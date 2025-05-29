from __future__ import annotations
from logging import Logger
import os
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from prog_policies.base import BaseDSL, BaseEnvironment

from ..utils import init_gru
from .base_vae import BaseVAE, ModelReturn, ProgramSequenceEncoder, ProgramSequenceDecoder, TrajectorySequenceDecoder
from .leaps_vq_vae import VectorQuantizer


class Accuracies(NamedTuple):
    progs_t: torch.Tensor
    progs_s: torch.Tensor
    a_h_t: torch.Tensor
    a_h_s: torch.Tensor
    struct_t: torch.Tensor
    struct_s: torch.Tensor


class Losses(NamedTuple):
    progs_rec: torch.Tensor
    a_h_rec: torch.Tensor
    a_h_disc: torch.Tensor
    struct_rec: torch.Tensor
    struct_disc: torch.Tensor
    latent: torch.Tensor


class DSSVAE(BaseVAE):
    """Disentangled Syntax and Semantic model.
    """    
    def __init__(self, dsl: BaseDSL, device: torch.device, env_cls: type[BaseEnvironment],
                 env_args: dict, max_program_length = 45, max_demo_length = 100, model_seed = 1,
                 hidden_size = 256, logger: Logger = None, name: str = None):
        super().__init__(dsl, device, env_cls, env_args, max_program_length, max_demo_length,
                         model_seed, hidden_size, logger=logger, name=name)
        
        self.syn_latent_size = self.hidden_size
        self.sem_latent_size = self.hidden_size
        
        self.vq_beta = 0.25
        
        # Inputs: enc(rho_i) (T). Output: enc_state (Z). Hidden state: h_i: z = h_t (Z).
        encoder_gru = nn.GRU(self.num_program_tokens, self.syn_latent_size + self.sem_latent_size)
        init_gru(encoder_gru)
        
        # Input: rho_i (T). Output: enc(rho_i) (T).
        token_encoder = nn.Embedding(self.num_program_tokens, self.num_program_tokens)
        
        self.prog_encoder = ProgramSequenceEncoder(token_encoder, encoder_gru)
        
        # Inputs: enc(rho_i) (T), z (Z). Output: dec_state (Z). Hidden state: h_i: h_0 = z (Z).
        decoder_gru = nn.GRU(self.syn_latent_size + self.sem_latent_size + self.num_program_tokens,
                             self.syn_latent_size + self.sem_latent_size)
        init_gru(decoder_gru)
        
        # Input: dec_state (Z), z (Z), enc(rho_i) (T). Output: prob(rho_hat) (T).
        decoder_mlp = nn.Sequential(
            self.init_(nn.Linear(self.syn_latent_size + self.sem_latent_size + self.num_program_tokens + self.syn_latent_size + self.sem_latent_size,
                                 self.syn_latent_size + self.sem_latent_size)),
            nn.Tanh(), self.init_(nn.Linear(self.syn_latent_size + self.sem_latent_size, self.num_program_tokens))
        )
        
        self.prog_decoder = ProgramSequenceDecoder(token_encoder, decoder_gru, decoder_mlp,
                                                   self.device, dsl, max_program_length)
        
        traj_rec_gru = nn.GRU(self.sem_latent_size + self.num_agent_actions + self.perceptions_size, self.hidden_size)
        init_gru(traj_rec_gru)
        
        # Inputs: pol_out (Z). Output: prob(a_hat) (A).
        traj_rec_mlp = nn.Sequential(
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.num_agent_actions))
        )
        
        # Input: s_i (CxHxW). Output: enc(s_i) (Z).
        perc_encoder = nn.Sequential(
            self.init_(nn.Linear(self.perceptions_size, self.perceptions_size)), nn.ReLU()
        )
        
        # Input: a_i (A). Output: enc(a_i) (A).
        action_encoder = nn.Embedding(self.num_agent_actions, self.num_agent_actions)
        
        self.traj_rec = TrajectorySequenceDecoder(action_encoder, perc_encoder, traj_rec_gru,
                                                  traj_rec_mlp, self.device, max_demo_length,
                                                  self.num_agent_actions)
        
        traj_disc_gru = nn.GRU(self.syn_latent_size + self.num_agent_actions + self.perceptions_size, self.hidden_size)
        init_gru(traj_disc_gru)
        
        # Inputs: pol_out (Z). Output: prob(a_hat) (A).
        traj_disc_mlp = nn.Sequential(
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.num_agent_actions))
        )
        
        self.traj_disc = TrajectorySequenceDecoder(action_encoder, perc_encoder, traj_disc_gru,
                                                   traj_disc_mlp, self.device, max_demo_length,
                                                   self.num_agent_actions)
        
        structure_dsl = dsl.structure_only()
        self.num_structure_tokens = len(structure_dsl.get_tokens())
        
        struct_rec_gru = nn.GRU(self.syn_latent_size + self.num_structure_tokens, self.hidden_size)
        init_gru(struct_rec_gru)
        
        struct_rec_mlp = nn.Sequential(
            self.init_(nn.Linear(self.hidden_size + self.num_structure_tokens + self.syn_latent_size,
                                 self.hidden_size)),
            nn.Tanh(), self.init_(nn.Linear(self.hidden_size, self.num_structure_tokens))
        )
        
        structure_token_encoder = nn.Embedding(self.num_structure_tokens, self.num_structure_tokens)
        
        self.struct_rec = ProgramSequenceDecoder(structure_token_encoder, struct_rec_gru,
                                                 struct_rec_mlp, self.device, structure_dsl,
                                                 max_program_length, only_structure=True)
        
        struct_disc_gru = nn.GRU(self.sem_latent_size + self.num_structure_tokens, self.hidden_size)
        init_gru(struct_disc_gru)
        
        struct_disc_mlp = nn.Sequential(
            self.init_(nn.Linear(self.hidden_size + self.num_structure_tokens + self.sem_latent_size,
                                 self.hidden_size)),
            nn.Tanh(), self.init_(nn.Linear(self.hidden_size, self.num_structure_tokens))
        )
        
        self.struct_disc = ProgramSequenceDecoder(structure_token_encoder, struct_disc_gru,
                                                  struct_disc_mlp, self.device, structure_dsl,
                                                  max_program_length, only_structure=True)
        
        # Encoder VAE utils
        self.encoder_syn_mu = torch.nn.Linear(self.syn_latent_size, self.syn_latent_size)
        self.encoder_syn_log_sigma = torch.nn.Linear(self.syn_latent_size, self.syn_latent_size)

        self.encoder_sem_quantizer = VectorQuantizer(10000, self.hidden_size, 0.25)
        
        self.to(self.device)

    def sample_latent_vector(self, enc_hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Sampling z with reperameterization trick
        enc_hidden_syn, enc_hidden_sem = torch.split(
            enc_hidden_state, [self.syn_latent_size, self.sem_latent_size], dim=-1
        )
        
        syn_mu = self.encoder_syn_mu(enc_hidden_syn)
        syn_log_sigma = self.encoder_syn_log_sigma(enc_hidden_syn)
        syn_sigma = torch.exp(syn_log_sigma)
        syn_std_z = torch.randn(syn_sigma.size(), device=self.device)
        
        z_syn = syn_mu + syn_sigma * syn_std_z
        
        q_sem = self.encoder_sem_quantizer(enc_hidden_sem)
        
        self.q_sem = q_sem
        self.z_sem = enc_hidden_sem
        
        self.z_syn_mu = syn_mu
        self.z_syn_sigma = syn_sigma
        
        return z_syn, q_sem
    
    def get_latent_loss(self):
        syn_mean_sq = self.z_syn_mu * self.z_syn_mu
        syn_stddev_sq = self.z_syn_sigma * self.z_syn_sigma
        vae_loss = 0.5 * torch.mean(syn_mean_sq + syn_stddev_sq - torch.log(syn_stddev_sq) - 1)
        
        commitment_loss = F.mse_loss(self.q_sem, self.z_sem.detach())
        embedding_loss = F.mse_loss(self.q_sem.detach(), self.z_sem)
        vq_loss = commitment_loss + self.vq_beta * embedding_loss
        return vae_loss + vq_loss
    
    def forward(self, data_batch: tuple, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True, adversarial_pass = False) -> tuple:
        perc_h, a_h, a_h_masks, progs, progs_masks, structs = data_batch
        
        enc_hidden_state = self.prog_encoder(progs, progs_masks)
        
        z_syn, z_sem = self.sample_latent_vector(enc_hidden_state)
        
        z_cat = torch.cat([z_syn, z_sem], dim=-1)
        
        if adversarial_pass:
            
            disc_a_h, disc_a_h_logits, disc_a_h_masks = self.traj_disc(
                z_syn.detach(), perc_h, a_h, a_h_masks, a_h_teacher_enforcing, True
            )
            
            disc_struct, disc_struct_logits, disc_struct_masks = self.struct_disc(
                z_sem.detach(), structs, progs_masks, prog_teacher_enforcing
            )
            
            return (z_syn, z_sem, None, None, None,
                    None, None, None,
                    disc_a_h, disc_a_h_logits, disc_a_h_masks,
                    None, None, None,
                    disc_struct, disc_struct_logits, disc_struct_masks)
        
        else:

            pred_progs, pred_progs_logits, pred_progs_masks = self.prog_decoder(
                z_cat, progs, progs_masks, prog_teacher_enforcing
            )
            
            rec_a_h, rec_a_h_logits, rec_a_h_masks = self.traj_rec(
                z_sem, perc_h, a_h, a_h_masks, a_h_teacher_enforcing, True
            )
            
            disc_a_h, disc_a_h_logits, disc_a_h_masks = self.traj_disc(
                z_syn, perc_h, a_h, a_h_masks, a_h_teacher_enforcing, True
            )

            rec_struct, rec_struct_logits, rec_struct_masks = self.struct_rec(
                z_syn, structs, progs_masks, prog_teacher_enforcing
            )
            
            disc_struct, disc_struct_logits, disc_struct_masks = self.struct_disc(
                z_sem, structs, progs_masks, prog_teacher_enforcing
            )
            
            return (z_syn, z_sem, pred_progs, pred_progs_logits, pred_progs_masks,
                    rec_a_h, rec_a_h_logits, rec_a_h_masks,
                    disc_a_h, disc_a_h_logits, disc_a_h_masks,
                    rec_struct, rec_struct_logits, rec_struct_masks,
                    disc_struct, disc_struct_logits, disc_struct_masks)
        
    def encode_program(self, prog: torch.Tensor):
        if prog.dim() == 1:
            prog = prog.unsqueeze(0)
        
        prog_mask = (prog != self.num_program_tokens - 1)
        
        z = self.encode(prog, prog_mask)
        
        return z
    
    def decode_vector(self, z: torch.Tensor):
        pred_progs, _, pred_progs_masks = self.decode(z, None, None, False)
        
        pred_progs_tokens = []
        for prog, prog_mask in zip(pred_progs, pred_progs_masks):
            pred_progs_tokens.append([0] + prog[prog_mask].cpu().numpy().tolist())
        
        return pred_progs_tokens
    
    def get_losses_and_accs(self, data_batch: tuple, output: ModelReturn,
                                  loss_fn: nn.CrossEntropyLoss) -> tuple[Losses, Accuracies]:
        _, a_h, a_h_masks, progs, progs_masks, structs = data_batch

        _, _, pred_progs, pred_progs_logits, pred_progs_masks, \
            rec_a_h, rec_a_h_logits, rec_a_h_masks, \
            disc_a_h, disc_a_h_logits, disc_a_h_masks, \
            rec_struct, rec_struct_logits, rec_struct_masks, \
            disc_struct, disc_struct_logits, disc_struct_masks = output
        
        if type(a_h) == torch.Tensor:
            # Combine first 2 dimensions of a_h (batch_size and demos_per_program)
            a_h = a_h.view(-1, a_h.shape[-1])
            a_h_masks = a_h_masks.view(-1, a_h_masks.shape[-1])
            
            # Skip first token in ground truth sequences
            a_h = a_h[:, 1:].contiguous()
            a_h_masks = a_h_masks[:, 1:].contiguous()
            
            # Flatten everything for loss calculation
            a_h_flat = a_h.view(-1, 1)
            a_h_masks_flat = a_h_masks.view(-1, 1)
        
        if type(progs) == torch.Tensor:
            # Combine first 2 dimensions of progs (batch_size and demos_per_program)
            progs = progs.view(-1, progs.shape[-1])
            progs_masks = progs_masks.view(-1, progs_masks.shape[-1])
            
            # Skip first token in ground truth sequences
            progs = progs[:, 1:].contiguous()
            progs_masks = progs_masks[:, 1:].contiguous()

            # Flatten everything for loss calculation
            progs_flat = progs.view(-1, 1)
            progs_masks_flat = progs_masks.view(-1, 1)
            
        if type(structs) == torch.Tensor:
            # Combine first 2 dimensions of structs (batch_size and demos_per_program)
            structs = structs.view(-1, structs.shape[-1])
            
            # Skip first token in ground truth sequences
            structs = structs[:, 1:].contiguous()

            # Flatten everything for loss calculation
            structs_flat = structs.view(-1, 1)
        
        if rec_a_h is not None:
            rec_a_h_logits = rec_a_h_logits.view(-1, rec_a_h_logits.shape[-1])
            rec_a_h_masks_flat = rec_a_h_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            rec_a_h_masks_flat_combined = torch.max(a_h_masks_flat, rec_a_h_masks_flat).squeeze()
        
        if rec_struct is not None:
            rec_struct_logits = rec_struct_logits.view(-1, rec_struct_logits.shape[-1])
            rec_struct_masks_flat = rec_struct_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            rec_struct_masks_flat_combined = torch.max(progs_masks_flat, rec_struct_masks_flat).squeeze()
        
        if disc_a_h is not None:
            disc_a_h_logits = disc_a_h_logits.view(-1, disc_a_h_logits.shape[-1])
            disc_a_h_masks_flat = disc_a_h_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            disc_a_h_masks_flat_combined = torch.max(a_h_masks_flat, disc_a_h_masks_flat).squeeze()
            
        if disc_struct is not None:
            disc_struct_logits = disc_struct_logits.view(-1, disc_struct_logits.shape[-1])
            disc_struct_masks_flat = disc_struct_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            disc_struct_masks_flat_combined = torch.max(progs_masks_flat, disc_struct_masks_flat).squeeze()
        
        if pred_progs is not None:
            pred_progs_logits = pred_progs_logits.view(-1, pred_progs_logits.shape[-1])
            pred_progs_masks_flat = pred_progs_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            progs_masks_flat_combined = torch.max(progs_masks_flat, pred_progs_masks_flat).squeeze()
        
        zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
        progs_loss, rec_a_h_loss, rec_struct_loss, disc_a_h_loss, disc_struct_loss = zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor
        
        if rec_a_h is not None:
            rec_a_h_loss = loss_fn(rec_a_h_logits[rec_a_h_masks_flat_combined],
                                   a_h_flat[rec_a_h_masks_flat_combined].view(-1))
            
        if rec_struct is not None:
            rec_struct_loss = loss_fn(rec_struct_logits[rec_struct_masks_flat_combined],
                                      structs_flat[rec_struct_masks_flat_combined].view(-1))
            
        if disc_a_h is not None:
            disc_a_h_loss = loss_fn(disc_a_h_logits[disc_a_h_masks_flat_combined],
                                    a_h_flat[disc_a_h_masks_flat_combined].view(-1))
            
        if disc_struct is not None:
            disc_struct_loss = loss_fn(disc_struct_logits[disc_struct_masks_flat_combined],
                                       structs_flat[disc_struct_masks_flat_combined].view(-1))
        
        if pred_progs is not None:
            progs_loss = loss_fn(pred_progs_logits[progs_masks_flat_combined],
                                 progs_flat[progs_masks_flat_combined].view(-1))

        latent_loss = self.get_latent_loss()
        
        with torch.no_grad():
            progs_s_accuracy, progs_t_accuracy = zero_tensor, zero_tensor
            if pred_progs is not None:
                progs_masks_combined = torch.max(progs_masks, pred_progs_masks)
                progs_t_accuracy = (pred_progs[progs_masks_combined] == progs[progs_masks_combined]).float().mean()
                progs_s_accuracy = (progs == pred_progs).min(dim=1).values.float().mean()
            
            a_h_s_accuracy, a_h_t_accuracy = zero_tensor, zero_tensor
            if rec_a_h is not None:
                a_h_masks_combined = torch.max(a_h_masks, rec_a_h_masks)
                a_h_t_accuracy = (rec_a_h[a_h_masks_combined] == a_h[a_h_masks_combined]).float().mean()
                a_h_s_accuracy = (a_h == rec_a_h).min(dim=1).values.float().mean()
                
            struct_s_accuracy, struct_t_accuracy = zero_tensor, zero_tensor
            if rec_struct is not None:
                struct_masks_combined = torch.max(progs_masks, rec_struct_masks)
                struct_t_accuracy = (rec_struct[struct_masks_combined] == structs[struct_masks_combined]).float().mean()
                struct_s_accuracy = (structs == rec_struct).min(dim=1).values.float().mean()
        
        return Losses(progs_loss, rec_a_h_loss, disc_a_h_loss, rec_struct_loss, disc_struct_loss, latent_loss),\
            Accuracies(progs_t_accuracy, progs_s_accuracy, a_h_t_accuracy, a_h_s_accuracy, struct_t_accuracy, struct_s_accuracy)

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
            prog_loss_coef: float = 1.0, a_h_loss_coef: float = 1.0,
            latent_loss_coef: float = 0.1,
            disable_prog_teacher_enforcing: bool = False,
            disable_a_h_teacher_enforcing: bool = False,
            optim_lr: float = 5e-4, save_params_each_epoch: bool = False,
            num_epochs: int = 100, logger: Logger = None,
            base_output_folder: str = 'output'):
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=optim_lr
        )
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        output_folder = os.path.join(base_output_folder, 'trainer', self.name)
        model_folder = os.path.join(output_folder, 'model')
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)
        
        if val_dataloader is not None:
            best_val_return = float('inf')
        
        training_info_csv = os.path.join(output_folder, 'training_info.csv')
        validation_info_csv = os.path.join(output_folder, 'validation_info.csv')
        
        fields = ['total_loss'] + [loss + '_loss' for loss in Losses._fields] +\
            [acc + '_accuracy' for acc in Accuracies._fields]
        
        with open(training_info_csv, 'w') as f:
            f.write('epoch,')
            f.write(','.join(fields))
            f.write('\n')
            
        if val_dataloader is not None:
            with open(validation_info_csv, 'w') as f:
                f.write('epoch,')
                f.write(','.join(fields))
                f.write('\n')
        
        for epoch in range(1, num_epochs + 1):
            self.log(f'Training epoch {epoch}/{num_epochs}')
            
            self.train()
            torch.set_grad_enabled(True)
            
            train_batches_total_losses = np.zeros((len(train_dataloader), 1))
            train_batches_losses = np.zeros((len(train_dataloader), len(Losses._fields)))
            train_batches_accs = np.zeros((len(train_dataloader), len(Accuracies._fields)))
            
            # Training step
            for i, train_batch in enumerate(train_dataloader):
                
                optimizer.zero_grad()
                
                adv_output = self(train_batch,
                                  not disable_prog_teacher_enforcing, 
                                  not disable_a_h_teacher_enforcing,
                                  adversarial_pass=True)
                adv_losses, adv_accs = self.get_losses_and_accs(train_batch, adv_output, loss_fn)
                
                total_adv_loss = adv_losses.a_h_disc + adv_losses.struct_disc
                
                total_adv_loss.backward()
                optimizer.zero_grad()

                output = self(train_batch,
                              not disable_prog_teacher_enforcing,
                              not disable_a_h_teacher_enforcing)
                losses, accs = self.get_losses_and_accs(train_batch, output, loss_fn)
                
                total_loss = prog_loss_coef * losses.progs_rec -\
                    losses.a_h_disc + losses.a_h_rec + losses.struct_rec - losses.struct_disc +\
                    latent_loss_coef * losses.latent
                
                total_loss.backward()
                optimizer.step()
                
                train_batches_total_losses[i] = total_loss.item()
                train_batches_losses[i] = np.array([loss.item() for loss in losses])
                train_batches_accs[i] = np.array([acc.item() for acc in accs])
                
            train_mean_total_loss = train_batches_total_losses.mean()
            train_mean_losses = train_batches_losses.mean(axis=0)
            train_mean_accs = train_batches_accs.mean(axis=0)
            
            self.log('Total loss: ' + str(train_mean_total_loss))
            self.log(str(Losses(*train_mean_losses.tolist())))
            self.log(str(Accuracies(*train_mean_accs.tolist())))
            
            with open(training_info_csv, 'a') as f:
                f.write(f'{epoch},{train_mean_total_loss},')
                f.write(','.join([str(loss) for loss in train_mean_losses]))
                f.write(',')
                f.write(','.join([str(acc) for acc in train_mean_accs]))
                f.write('\n')
                
            if val_dataloader is not None:
                self.log(f'Validation epoch {epoch}/{num_epochs}')
                
                self.eval()
                torch.set_grad_enabled(False)
                
                val_batches_total_losses = np.zeros((len(val_dataloader), 1))
                val_batches_losses = np.zeros((len(val_dataloader), len(Losses._fields)))
                val_batches_accs = np.zeros((len(val_dataloader), len(Accuracies._fields)))
                
                # Validation step
                for i, val_batch in enumerate(val_dataloader):
                    
                    output = self(val_batch,
                                  not disable_prog_teacher_enforcing,
                                  not disable_a_h_teacher_enforcing)
                    losses, accs = self.get_losses_and_accs(val_batch, output, loss_fn)
                    
                    total_loss = prog_loss_coef * losses.progs_rec -\
                        losses.a_h_disc + losses.a_h_rec + losses.struct_rec - losses.struct_disc +\
                        latent_loss_coef * losses.latent
                    
                    val_batches_total_losses[i] = total_loss.item()
                    val_batches_losses[i] = np.array([loss.item() for loss in losses])
                    val_batches_accs[i] = np.array([acc.item() for acc in accs])

                val_mean_total_loss = val_batches_total_losses.mean()
                val_mean_losses = val_batches_losses.mean(axis=0)
                val_mean_accs = val_batches_accs.mean(axis=0)

                self.log('Total loss: ' + str(val_mean_total_loss))
                self.log(str(Losses(*val_mean_losses.tolist())))
                self.log(str(Accuracies(*val_mean_accs.tolist())))
                    
                with open(validation_info_csv, 'a') as f:
                    f.write(f'{epoch},{val_mean_total_loss},')
                    f.write(','.join([str(loss) for loss in val_mean_losses]))
                    f.write(',')
                    f.write(','.join([str(acc) for acc in val_mean_accs]))
                    f.write('\n')
                    
                if val_mean_total_loss < best_val_return:
                    self.log(f'New best validation loss: {val_mean_total_loss}')
                    best_val_return = val_mean_total_loss
                    params_path = os.path.join(model_folder, f'best_val.ptp')
                    torch.save(self.state_dict(), params_path)
                    self.log(f'Parameters saved to {params_path}')
