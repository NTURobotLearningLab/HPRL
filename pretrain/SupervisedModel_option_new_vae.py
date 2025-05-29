import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch import linalg as LA

import wandb

from pretrain.BaseModel import BaseModel
from pretrain.models_option_new_vae import VAE, ConditionPolicy, ProgramVAE
from rl.utils import masked_mean
import torch.nn.functional as F
from pretrain.utils import analyze_z_bz

# Add global step tracker as a variable in the module scope
# But we'll also track it as a class attribute in the SupervisedModel

def calculate_accuracy(logits, targets, mask, batch_shape):
    masked_preds = (logits.argmax(dim=-1, keepdim=True) * mask).view(*batch_shape, 1)
    masked_targets = (targets * mask).view(*batch_shape, 1)
    t_accuracy = 100 * masked_mean((masked_preds == masked_targets).float(), mask.view(*masked_preds.shape),
                                   dim=1).mean()

    p_accuracy = 100 * (masked_preds.squeeze() == masked_targets.squeeze()).all(dim=1).float().mean()
    return t_accuracy, p_accuracy
    # here t_accuracy is the total tokens accuracy (allowing errors in one program sequence)
    # p_accuracy is the program accuracy (all tokens in one program sequence must be correct)


class SupervisedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(SupervisedModel, self).__init__(ProgramVAE, *args, **kwargs)
        self._two_head = self.config['two_head']
        self.max_demo_length = self.config['max_demo_length']
        self.condition_loss_coef = self.config['loss']['condition_loss_coef']
        self._vanilla_ae = self.config['AE']
        self._disable_decoder = self.config['net']['decoder']['freeze_params']
        self._disable_condition = self.config['net']['condition']['freeze_params']
        self.condition_states_source = self.config['net']['condition']['observations']
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        
        # Add eval metrics storage for averaging
        self.eval_metrics = {}
        
        # Add global step counter as a class attribute
        self.global_train_step = 0
        
        print("SupervisedModel: self.net.vae.decoder.setup: ", self.net.vae.decoder.setup)
        # debug code
        self._debug = self.config['debug']
        if self._debug:
            self.debug_dict_act = {}
            self.debug_dict_eval_act = {}

    @property
    def is_recurrent(self):
        return self.net.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.net.base.recurrent_hidden_state_size

    def _get_hinge_loss(self, z, b_z, margin=1.0):
        """
        Contrastive loss with 1:1 matching.
        - z[i] should be close to b_z[i] (positive pair)
        - z[i] should be at least `margin` away from b_z[j] where j ≠ i (negative pairs)

        :param z:   Program embeddings, shape (B, 64)
        :param b_z: Behavior embeddings, shape (B, 64)
        :param margin: Margin for the negative pairs

        :return: Scalar contrastive loss
        """
        B, D = z.shape
        # Compute pairwise L2 distances between z and b_z => shape (B, B)
        dist_matrix = torch.cdist(z, b_z, p=2)  # dist_matrix[i, j] = ||z[i] - b_z[j]||
        # print(f"dist_matrix_shape: {dist_matrix.shape}")
        # print(f"dist_matrix: {dist_matrix}")

        # Positive loss: distance between z[i] and b_z[i]
        pos_loss = torch.diagonal(dist_matrix).sum() / B

        # Negative loss: hinge loss on margin - distance for all i ≠ j
        mask = ~torch.eye(B, dtype=torch.bool, device=z.device)  # mask to exclude diagonal
        neg_dists = dist_matrix[mask].view(B, B - 1)
        neg_loss = F.relu(margin - neg_dists).sum() / (B * (B - 1))

        return pos_loss + neg_loss
    def _get_mse_loss(self, z, b_z):
        loss = F.mse_loss(z, b_z)
        return loss

    def _get_clip_loss(self, z, b_z):
        """
        CLIP loss. Adapted from 
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
        
        :param z:   Program embeddings, shape (B, D)
        :param b_z: Behavior embeddings, shape (B, D)
        
        :return: Scalar contrastive loss
        """
        # Normalize embeddings
        z = z / torch.norm(z, dim=-1, keepdim=True)
        b_z = b_z / torch.norm(b_z, dim=-1, keepdim=True)
        # Compute similarity matrix (cosine similarity)
        similarity = torch.matmul(z, b_z.T)
        
        # Temperature parameter (can be made configurable)
        similarity = similarity * self.logit_scale.exp()
        
        # CLIP loss - both directions (z->b_z and b_z->z)
        labels = torch.arange(len(similarity), device=similarity.device)
        loss_z = F.cross_entropy(similarity, labels)
        loss_b_z = F.cross_entropy(similarity.T, labels)
        clip_loss = (loss_z + loss_b_z) / 2.0

        # Top-1 accuracy (both directions)
        pred_z = similarity.argmax(dim=1)       # z -> b_z
        pred_b_z = similarity.T.argmax(dim=1)   # b_z -> z
        acc_z = (pred_z == labels).float().mean()
        acc_b_z = (pred_b_z == labels).float().mean()
        clip_acc = (acc_z + acc_b_z) / 2.0

        return clip_loss, clip_acc

    def _get_condition_loss(self, a_h, a_h_len, action_logits, action_masks):
        """ loss between ground truth trajectories and predicted action sequences

        :param a_h(int16): B x num_demo_per_program x max_demo_length
        :param a_h_len(int16): a_h_len: B x num_demo_per_program
        :param action_logits: (B * num_demo_per_programs) x max_a_h_len x num_actions
        :param action_masks: (B * num_demo_per_programs) x max_a_h_len x 1
        :return (float): condition policy loss
        """
        batch_size_x_num_demo_per_program, max_a_h_len, num_actions = action_logits.shape
        assert max_a_h_len == a_h.shape[-1]

        padded_preds = action_logits

        """ add dummy logits to targets """
        target_masks = a_h != self.net.condition_policy.num_agent_actions - 1
        # remove temporarily added no-op actions in case of empty trajectory to
        # verify target masks
        a_h_len2 = a_h_len - (a_h[:,:,0] == self.net.condition_policy.num_agent_actions - 1).to(a_h_len.dtype)
        assert (target_masks.sum(dim=-1).squeeze() == a_h_len2.squeeze()).all()
        targets = torch.where(target_masks, a_h, (num_actions-1) * torch.ones_like(a_h))

        """ condition mask """
        # flatten everything and select actions that you want in backpropagation
        target_masks = target_masks.view(-1, 1)
        action_masks = action_masks.view(-1, 1)
        cond_mask = torch.max(action_masks, target_masks)

        # gather prediction that needs backpropagation
        subsampled_targets = targets.view(-1,1)[cond_mask].long()
        subsampled_padded_preds = padded_preds.view(-1, num_actions)[cond_mask.squeeze()]

        condition_loss = self.loss_fn(subsampled_padded_preds, subsampled_targets)

        """ calculate accuracy """
        with torch.no_grad():
            batch_shape = padded_preds.shape[:-1]
            cond_t_accuracy, cond_p_accuracy = calculate_accuracy(padded_preds.view(-1, num_actions),
                                                                  targets.view(-1, 1), cond_mask, batch_shape)

        return condition_loss, cond_t_accuracy, cond_p_accuracy

    def _greedy_rollout(self, batch, z, targets, trg_mask, mode):
        # autoregressive action predictions of decoder and condition policy
        programs, _, _, s_h, s_h_len, a_h, a_h_len = batch

        if mode == 'train' and self.condition_states_source != 'initial_state':
            zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
            return (zero_tensor, zero_tensor, zero_tensor, zero_tensor), None, None

        with torch.no_grad():
            # greedy rollout of decoder
            greedy_outputs = self.net.vae.decoder(programs, z, teacher_enforcing=False, deterministic=True)
            _, z_generated_programs, _, _, greedy_output_logits, _, _, pred_program_masks, _ = greedy_outputs
            # print(f"z_generated_programs: {z_generated_programs}")
            z_generated_programs_str = [self.dsl.intseq2str(prg) for prg in z_generated_programs]
            """ calculate accuracy """
            logits = greedy_output_logits.view(-1, greedy_output_logits.shape[-1])
            pred_mask = pred_program_masks.view(-1, 1)
            vae_mask = torch.max(pred_mask, trg_mask)
            with torch.no_grad():
                batch_shape = greedy_output_logits.shape[:-1]
                greedy_t_accuracy, greedy_p_accuracy = calculate_accuracy(logits, targets, vae_mask, batch_shape)

            init_states = s_h[0][:, :, 0, :, :, :].unsqueeze(2)
            _, _, _, action_logits, action_masks, _ = self.net.condition_policy(init_states, a_h, z, teacher_enforcing=False, deterministic=True)
            _, greedy_a_accuracy, greedy_d_accuracy = self._get_condition_loss(a_h, a_h_len, action_logits,
                                                                               action_masks)

            # 2 random vectors
            generated_programs = None
            if mode == 'eval':
                rand_z = torch.randn((2, z.shape[1])).to(z.dtype).to(z.device)
                generated_outputs = self.net.vae.decoder(None, rand_z, teacher_enforcing=False, deterministic=True)
                generated_programs = [self.dsl.intseq2str(prg) for prg in generated_outputs[1]]

        return (greedy_t_accuracy, greedy_p_accuracy, greedy_a_accuracy, greedy_d_accuracy), z_generated_programs_str, logits


    def _run_batch(self, batch, mode='train'):
        """ training on one batch

        :param batch: list of 6 elements:
                      programs(long): ground truth programs: B x max_program_len
                      ids(str): program ids
                      trg_mask(bool): masks for identifying valid tokens in programs: B x max_program_len x 1
                      s_h(bool): B x num_demo_per_program x max_demo_length(currently 1) x C x W x H
                      a_h(int16): B x num_demo_per_program x max_demo_length
                      a_h_len(int16): B x num_demo_per_program

        :param mode(str): execution mode, train or eval
        :return (dict): batch_info containing accuracy, loss and predicitons
        """
        # Use the class attribute instead of the global variable
        
        # Do mode-based setup
        if mode == 'train':
            self.net.train()
            torch.set_grad_enabled(True)
        elif mode == 'eval':
            self.net.eval()
            torch.set_grad_enabled(False)
            
            # Clear eval metrics storage at the start of a new evaluation
            if not self.eval_metrics:
                self.eval_metrics = {
                    'loss/total': [],
                    'loss/z_rec': [],
                    'loss/b_z_rec': [],
                    'loss/z_lat': [],
                    'loss/b_z_lat': [],
                    'loss/comb_lat': [],
                    'loss/z_condition': [],
                    'loss/b_z_condition': [],
                    'loss/clip': [],
                    'loss/clip_accuracy': [],
                    'loss/contrastive': [],
                    'loss/mse': [],
                    'loss/cosine': [],
                    'loss/l2': [],
                    'loss/l3': [],
                    'z_vs_b/decoder_token_accuracy/z': [],
                    'z_vs_b/decoder_token_accuracy/b_z': [],
                    'z_vs_b/decoder_program_accuracy/z': [],
                    'z_vs_b/decoder_program_accuracy/b_z': [],
                    'z_vs_b/condition_action_accuracy/z': [],
                    'z_vs_b/condition_action_accuracy/b_z': [],
                    'z_vs_b/condition_demo_accuracy/z': [],
                    'z_vs_b/condition_demo_accuracy/b_z': [],
                    'z_vs_b/decoder_greedy_token_accuracy/z': [],
                    'z_vs_b/decoder_greedy_token_accuracy/b_z': [],
                    'z_vs_b/decoder_greedy_program_accuracy/z': [],
                    'z_vs_b/decoder_greedy_program_accuracy/b_z': [],
                    'z_vs_b/condition_greedy_action_accuracy/z': [],
                    'z_vs_b/condition_greedy_action_accuracy/b_z': [],
                    'z_vs_b/condition_greedy_demo_accuracy/z': [],
                    'z_vs_b/condition_greedy_demo_accuracy/b_z': [],
                    'z_vs_b/bz_norm_mean': [],
                    'z_vs_b/z_norm_mean': [],
                    'z_vs_b/norm_ratio': [],
                    'z_vs_b/z_bz_angle': [],
                }

        programs, ids, trg_mask, s_h, s_h_len, a_h, a_h_len = batch

        """ forward pass """
        z_output, b_z_output, encoder_time, decoder_time = self.net(programs, trg_mask, s_h, a_h, s_h_len, a_h_len, deterministic=True)
        z_pred_programs       = z_output['pred_programs']
        z_pred_programs_len   = z_output['pred_programs_len']
        z_output_logits       = z_output['output_logits']
        z_eop_pred_programs   = z_output['eop_pred_programs']
        z_eop_output_logits   = z_output['eop_output_logits']
        z_pred_program_masks  = z_output['pred_program_masks']
        z_action_logits       = z_output['action_logits']
        z_action_masks        = z_output['action_masks']
        pre_tanh_z            = z_output['pre_tanh']
        z                     = z_output['z']
        z_mu = z_output['z_mu']

        b_z_pred_programs       = b_z_output['pred_programs']
        b_z_pred_programs_len   = b_z_output['pred_programs_len']
        b_z_output_logits       = b_z_output['output_logits']
        b_z_eop_pred_programs   = b_z_output['eop_pred_programs']
        b_z_eop_output_logits   = b_z_output['eop_output_logits']
        b_z_pred_program_masks  = b_z_output['pred_program_masks']
        b_z_action_logits       = b_z_output['action_logits']
        b_z_action_masks        = b_z_output['action_masks']
        pre_tanh_b_z            = b_z_output['pre_tanh']
        b_z                     = b_z_output['z']
        b_z_mu = b_z_output['z_mu']

        if self.config['normalize_latent']:
            z = z / torch.norm(z, dim=-1, keepdim=True)  # Normalize z
            b_z = b_z / torch.norm(b_z, dim=-1, keepdim=True)  # Normalize b_z
        zbz_analysis = analyze_z_bz(z, b_z)

        # calculate latent program embedding norm
        assert len(pre_tanh_z.shape) == 2
        batch_pre_tanh_z_inf_norm_mean = LA.vector_norm(pre_tanh_z.abs(), ord=float('inf'), dim=1).mean()
        batch_pre_tanh_z_mean_mean  = pre_tanh_z.abs().mean(dim=1).mean()
        batch_pre_tanh_z_outlier_ratio = (pre_tanh_z.abs() > 0.98).sum() / len(pre_tanh_z.flatten())


        """ flatten inputs and outputs for loss calculation """
        # skip first token DEF for loss calculation
        targets = programs[:, 1:].contiguous().view(-1, 1)
        trg_mask = trg_mask[:, 1:].contiguous().view(-1, 1)
        z_logits = z_output_logits.view(-1, z_output_logits.shape[-1])
        b_z_logits = b_z_output_logits.view(-1, b_z_output_logits.shape[-1])
        pred_mask = z_pred_program_masks.view(-1, 1)
        # need to penalize shorter and longer predicted programs
        vae_mask = torch.max(pred_mask, trg_mask)

        # Do backprop
        if mode == 'train':
            self.optimizer.zero_grad()

        zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
        z_lat_loss, bz_lat_loss, comb_lat_loss, rec_loss, z_condition_loss, b_z_condition_loss = zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor
        z_cond_t_accuracy, z_cond_p_accuracy, b_z_cond_t_accuracy, b_z_cond_p_accuracy = zero_tensor, zero_tensor, zero_tensor, zero_tensor
        if not self._disable_decoder:
            z_rec_loss = self.loss_fn(z_logits[vae_mask.squeeze()], (targets[vae_mask.squeeze()]).view(-1))
            b_z_rec_loss = self.loss_fn(b_z_logits[vae_mask.squeeze()], (targets[vae_mask.squeeze()]).view(-1))
        if not self._vanilla_ae:
            z_lat_loss = self.net.vae.latent_loss(self.net.vae.z_mean, self.net.vae.z_sigma)
            bz_lat_loss = self.net.vae.latent_loss(self.net.vae.b_z_mean, self.net.vae.b_z_sigma)
            comb_lat_loss = self.net.vae.latent_loss(self.net.vae.z_mean, self.net.vae.b_z_mean)
        if not self._disable_condition:
            z_condition_loss, z_cond_t_accuracy, z_cond_p_accuracy = self._get_condition_loss(a_h, a_h_len, z_action_logits,
                                                                                        z_action_masks)
            b_z_condition_loss, b_z_cond_t_accuracy, b_z_cond_p_accuracy = self._get_condition_loss(a_h, a_h_len,
                                                                                               b_z_action_logits,
                                                                                               b_z_action_masks)
        clip_loss, clip_acc = self._get_clip_loss(z_mu, b_z_mu)
        hinge_loss = self._get_hinge_loss(z_mu, b_z_mu, self.config['loss']['contrastive_loss_margin'])
        mse_loss = self._get_mse_loss(z_mu, b_z_mu)
        cosine_sim_loss = 1-F.cosine_similarity(z_mu, b_z_mu, dim=1).mean()
        l2_loss = torch.norm(z_mu - b_z_mu, p=2, dim=1).mean()
        l3_loss = torch.norm(z_mu - b_z_mu, p=3, dim=1).mean()

        # total loss
        cfg_losses = self.config['loss']['enabled_losses']
        loss = 0.0

        if not self.program_frozen and not self.start_decoder_finetune:
            if cfg_losses.get('z_rec', False):
                loss += z_rec_loss
            if cfg_losses.get('b_z_rec', False):
                loss += b_z_rec_loss
            if 'clip' in cfg_losses.get('contrastive_loss', []):
                loss += clip_loss
            if 'hinge' in cfg_losses.get('contrastive_loss', []):
                loss += hinge_loss
            if 'mse' in cfg_losses.get('contrastive_loss', []):
                loss += mse_loss
            if 'cosine' in cfg_losses.get('contrastive_loss', []):
                loss += cosine_sim_loss
            if 'l2' in cfg_losses.get('contrastive_loss', []):
                loss += l2_loss
            if 'l3' in cfg_losses.get('contrastive_loss', []):
                loss += l3_loss
            if cfg_losses.get('latent', False) != "none":
                if cfg_losses.get('z_latent', False) == 'separate':
                    loss += self.config['loss']['z_latent_loss_coef'] * z_lat_loss + self.config['loss']['bz_latent_loss_coef'] * bz_lat_loss
                elif cfg_losses.get('z_latent', False) == 'combined':
                    loss += self.config['loss']['z_latent_loss_coef'] * comb_lat_loss
            if cfg_losses.get('z_condition', False):
                loss += self.config['loss']['condition_loss_coef'] * z_condition_loss
            if cfg_losses.get('b_z_condition', False):
                loss += self.config['loss']['condition_loss_coef'] * b_z_condition_loss
        elif self.program_frozen:
            # If the program is frozen, we only compute the cosine loss
            loss += cosine_sim_loss
        elif self.start_decoder_finetune:
            loss += b_z_rec_loss

        # loss = contrastive_loss 

        if mode == 'train':
            loss.backward()
            self.optimizer.step()
            # Increment the global step only in training mode
            self.global_train_step += 1
            
        """ calculate accuracy """
        with torch.no_grad():
            batch_shape = z_output_logits.shape[:-1]
            z_t_accuracy, z_p_accuracy = calculate_accuracy(z_logits, targets, vae_mask, batch_shape)
            b_z_t_accuracy, b_z_p_accuracy = calculate_accuracy(b_z_logits, targets, vae_mask, batch_shape)
            z_greedy_accuracies, z_generated_programs, z_glogits = self._greedy_rollout(batch, z, targets, trg_mask, mode)
            b_z_greedy_accuracies, b_z_generated_programs, b_z_glogits = self._greedy_rollout(batch, b_z, targets, trg_mask, mode)
            z_greedy_t_accuracy, z_greedy_p_accuracy, z_greedy_a_accuracy, z_greedy_d_accuracy = z_greedy_accuracies
            b_z_greedy_t_accuracy, b_z_greedy_p_accuracy, b_z_greedy_a_accuracy, b_z_greedy_d_accuracy = b_z_greedy_accuracies
        
        if mode == 'train':
            # Log metrics with the global step
            wandb.log({
                f'{mode}/loss/total': loss.item(),
                f'{mode}/loss/z_rec': z_rec_loss.item(),
                f'{mode}/loss/b_z_rec': b_z_rec_loss.item(),
                f'{mode}/loss/z_lat': z_lat_loss.item(),
                f'{mode}/loss/b_z_lat':bz_lat_loss.item(),
                f'{mode}/loss/comb_lat': comb_lat_loss.item(),
                f'{mode}/loss/z_condition': z_condition_loss.item(),
                f'{mode}/loss/b_z_condition': b_z_condition_loss.item(),
                f'{mode}/loss/clip': clip_loss.item(),
                f'{mode}/loss/clip_accuracy': clip_acc.item(),
                f'{mode}/loss/contrastive': hinge_loss.item(),
                f'{mode}/loss/mse': mse_loss.item(),
                f'{mode}/loss/cosine': cosine_sim_loss.item(),
                f'{mode}/loss/l2': l2_loss.item(),
                f'{mode}/loss/l3': l3_loss.item(),
                f'{mode}/z_vs_b/decoder_token_accuracy': {
                    'z': z_t_accuracy.item(),
                    'b_z': b_z_t_accuracy.item()
                },
                f'{mode}/z_vs_b/decoder_program_accuracy': {
                    'z': z_p_accuracy.item(),
                    'b_z': b_z_p_accuracy.item()
                },
                f'{mode}/z_vs_b/condition_action_accuracy': {
                    'z': z_cond_t_accuracy.item(),
                    'b_z': b_z_cond_t_accuracy.item()
                },
                f'{mode}/z_vs_b/condition_demo_accuracy': {
                    'z': z_cond_p_accuracy.item(),
                    'b_z': b_z_cond_p_accuracy.item()
                },
                f'{mode}/z_vs_b/z_norm_mean': zbz_analysis['z_norm'].item(),
                f'{mode}/z_vs_b/bz_norm_mean': zbz_analysis['bz_norm'].item(),
                f'{mode}/z_vs_b/norm_ratio': zbz_analysis['scale_ratio'].item(),
                f'{mode}/z_vs_b/z_bz_angle': zbz_analysis['angle_deg'],
            }, step=self.global_train_step)
            
        elif mode == "eval":
            # Store metrics for averaging later
            self.eval_metrics['loss/total'].append(loss.item())
            self.eval_metrics['loss/z_rec'].append(z_rec_loss.item())
            self.eval_metrics['loss/b_z_rec'].append(b_z_rec_loss.item())
            self.eval_metrics['loss/z_lat'].append(z_lat_loss.item())
            self.eval_metrics['loss/b_z_lat'].append(bz_lat_loss.item())
            self.eval_metrics['loss/comb_lat'].append(comb_lat_loss.item())
            self.eval_metrics['loss/z_condition'].append(z_condition_loss.item())
            self.eval_metrics['loss/b_z_condition'].append(b_z_condition_loss.item())
            self.eval_metrics['loss/clip'].append(clip_loss.item())
            self.eval_metrics['loss/clip_accuracy'].append(clip_acc.item())
            self.eval_metrics['loss/contrastive'].append(hinge_loss.item())
            self.eval_metrics['loss/mse'].append(mse_loss.item())
            self.eval_metrics['loss/cosine'].append(cosine_sim_loss.item())
            self.eval_metrics['loss/l2'].append(l2_loss.item())
            self.eval_metrics['loss/l3'].append(l3_loss.item())

            
            self.eval_metrics['z_vs_b/decoder_token_accuracy/z'].append(z_t_accuracy.item())
            self.eval_metrics['z_vs_b/decoder_token_accuracy/b_z'].append(b_z_t_accuracy.item())
            self.eval_metrics['z_vs_b/decoder_program_accuracy/z'].append(z_p_accuracy.item())
            self.eval_metrics['z_vs_b/decoder_program_accuracy/b_z'].append(b_z_p_accuracy.item())
            self.eval_metrics['z_vs_b/condition_action_accuracy/z'].append(z_cond_t_accuracy.item())
            self.eval_metrics['z_vs_b/condition_action_accuracy/b_z'].append(b_z_cond_t_accuracy.item())
            self.eval_metrics['z_vs_b/condition_demo_accuracy/z'].append(z_cond_p_accuracy.item())
            self.eval_metrics['z_vs_b/condition_demo_accuracy/b_z'].append(b_z_cond_p_accuracy.item())
            
            # Store greedy metrics if available
            self.eval_metrics['z_vs_b/decoder_greedy_token_accuracy/z'].append(z_greedy_t_accuracy.item())
            self.eval_metrics['z_vs_b/decoder_greedy_token_accuracy/b_z'].append(b_z_greedy_t_accuracy.item())
            self.eval_metrics['z_vs_b/decoder_greedy_program_accuracy/z'].append(z_greedy_p_accuracy.item())
            self.eval_metrics['z_vs_b/decoder_greedy_program_accuracy/b_z'].append(b_z_greedy_p_accuracy.item())
            self.eval_metrics['z_vs_b/condition_greedy_action_accuracy/z'].append(z_greedy_a_accuracy.item())
            self.eval_metrics['z_vs_b/condition_greedy_action_accuracy/b_z'].append(b_z_greedy_a_accuracy.item())
            self.eval_metrics['z_vs_b/condition_greedy_demo_accuracy/z'].append(z_greedy_d_accuracy.item())
            self.eval_metrics['z_vs_b/condition_greedy_demo_accuracy/b_z'].append(b_z_greedy_d_accuracy.item())
            self.eval_metrics['z_vs_b/z_norm_mean'].append(zbz_analysis['z_norm'])
            self.eval_metrics['z_vs_b/bz_norm_mean'].append(zbz_analysis['bz_norm'])
            self.eval_metrics['z_vs_b/norm_ratio'].append(zbz_analysis['scale_ratio'])
            self.eval_metrics['z_vs_b/z_bz_angle'].append(zbz_analysis['angle_deg'])


        batch_info = {
            'z_decoder_token_accuracy': z_t_accuracy.detach().cpu().numpy().item(),
            'z_decoder_program_accuracy': z_p_accuracy.detach().cpu().numpy().item(),
            'z_condition_action_accuracy': z_cond_t_accuracy.detach().cpu().numpy().item(),
            'z_condition_demo_accuracy': z_cond_p_accuracy.detach().cpu().numpy().item(),
            'z_decoder_greedy_token_accuracy': z_greedy_t_accuracy.detach().cpu().numpy().item(),
            'z_decoder_greedy_program_accuracy': z_greedy_p_accuracy.detach().cpu().numpy().item(),
            'z_condition_greedy_action_accuracy': z_greedy_a_accuracy.detach().cpu().numpy().item(),
            'z_condition_greedy_demo_accuracy': z_greedy_d_accuracy.detach().cpu().numpy().item(),
            'b_z_decoder_token_accuracy': b_z_t_accuracy.detach().cpu().numpy().item(),
            'b_z_decoder_program_accuracy': b_z_p_accuracy.detach().cpu().numpy().item(),
            'b_z_condition_action_accuracy': b_z_cond_t_accuracy.detach().cpu().numpy().item(),
            'b_z_condition_demo_accuracy': b_z_cond_p_accuracy.detach().cpu().numpy().item(),
            'b_z_decoder_greedy_token_accuracy': b_z_greedy_t_accuracy.detach().cpu().numpy().item(),
            'b_z_decoder_greedy_program_accuracy': b_z_greedy_p_accuracy.detach().cpu().numpy().item(),
            'b_z_condition_greedy_action_accuracy': b_z_greedy_a_accuracy.detach().cpu().numpy().item(),
            'b_z_condition_greedy_demo_accuracy': b_z_greedy_d_accuracy.detach().cpu().numpy().item(),
            'clip_loss_accuracy': clip_acc.detach().cpu().numpy().item(),
            'pre_tanh_z_inf_norm_mean': batch_pre_tanh_z_inf_norm_mean.detach().cpu().numpy().item(),
            'pre_tanh_z_mean_mean': batch_pre_tanh_z_mean_mean.detach().cpu().numpy().item(),
            'pre_tanh_z_outlier_ratio': batch_pre_tanh_z_outlier_ratio.detach().cpu().numpy().item(),
            'total_loss': loss.detach().cpu().numpy().item(),
            'z_rec_loss': z_rec_loss.detach().cpu().numpy().item(),
            'b_z_rec_loss': b_z_rec_loss.detach().cpu().numpy().item(),
            'z_lat_loss': z_lat_loss.detach().cpu().numpy().item(),
            'b_z_lat_loss': bz_lat_loss.detach().cpu().numpy().item(),
            'comb_lat_loss': comb_lat_loss.detach().cpu().numpy().item(),
            'z_condition_loss': z_condition_loss.detach().cpu().numpy().item(),
            'b_z_condition_loss': b_z_condition_loss.detach().cpu().numpy().item(),
            'clip_loss': clip_loss.detach().cpu().numpy().item(),
            'contrastive_loss': hinge_loss.detach().cpu().numpy().item(),
            'mse_loss': mse_loss.detach().cpu().numpy().item(),
            'cosine_loss': cosine_sim_loss.detach().cpu().numpy().item(),
            'l2_loss': l2_loss.detach().cpu().numpy().item(),
            'l3_loss': l3_loss.detach().cpu().numpy().item(),
            'gt_programs': programs.detach().cpu().numpy(),
            'z_pred_programs': z_pred_programs.detach().cpu().numpy(),
            'b_z_pred_programs': b_z_pred_programs.detach().cpu().numpy(),
            'z_generated_programs': z_generated_programs,
            'b_z_generated_programs': b_z_generated_programs,
            'program_ids': ids,
            'latent_vectors': z.detach().cpu().numpy().tolist(),
            'behavior_vectors': b_z.detach().cpu().numpy().tolist(),
            'encoder_time': encoder_time,
            'decoder_time': decoder_time,
            'zbz_bz_norm': zbz_analysis['bz_norm'].detach().cpu().numpy().item(),
            'zbz_z_norm': zbz_analysis['z_norm'].detach().cpu().numpy().item(),
            'zbz_scale_ratio': zbz_analysis['scale_ratio'].detach().cpu().numpy().item(),
            'zbz_angle_deg': zbz_analysis['angle_deg'],
            }

        return batch_info
