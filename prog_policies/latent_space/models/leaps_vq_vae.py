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

from .base_vae import BaseVAE, ModelReturn, ProgramSequenceEncoder, ProgramSequenceDecoder, TrajectorySequenceDecoder


class VectorQuantizer(nn.Module):
    
    def __init__(self, device: torch.device, num_embeddings: int, embedding_dim: int,
                 beta: float = 0.25):
        super().__init__()
        self.vq_loss = torch.tensor(0.0, device=device, requires_grad=True)
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.codebook = nn.Embedding(self.K, self.D)
        self.codebook.weight.data.normal_()
    
    def forward(self, latents: torch.Tensor):
        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.codebook.weight ** 2, dim=1) - \
               2 * torch.matmul(latents, self.codebook.weight.t())  # [B x K]
               
        # probs = F.softmax(-dist, dim=1)  # [B x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [B, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [B x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.codebook.weight)  # [B x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        self.vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents
    
    def get_loss(self):
        return self.vq_loss


class Accuracies(NamedTuple):
    progs_t: torch.Tensor
    progs_s: torch.Tensor
    a_h_t: torch.Tensor
    a_h_s: torch.Tensor


class Losses(NamedTuple):
    progs_rec: torch.Tensor
    a_h_rec: torch.Tensor
    latent: torch.Tensor


class LeapsVQVAE(BaseVAE):
    def __init__(self, dsl: BaseDSL, device: torch.device, env_cls: type[BaseEnvironment],
                 env_args: dict, max_program_length = 45, max_demo_length = 100, model_seed = 1,
                 hidden_size = 256, vq_pooling = False, vq_dim = 256, vq_size = 50000,
                 logger: Logger = None, name: str = None, wandb_args: dict = None):
        super().__init__(dsl, device, env_cls, env_args, max_program_length, max_demo_length,
                         model_seed, hidden_size, logger=logger, name=name, wandb_args=wandb_args)
        
        # Input: rho_i (T). Output: enc(rho_i) (T).
        token_encoder = nn.Embedding(self.num_program_tokens, self.num_program_tokens)
        
        self.prog_encoder = ProgramSequenceEncoder(
            token_encoder,
            gru=nn.GRU(self.num_program_tokens, self.hidden_size)
        )
        
        self.prog_decoder = ProgramSequenceDecoder(
            token_encoder, 
            gru=nn.GRU(self.hidden_size + self.num_program_tokens, self.hidden_size),
            mlp=nn.Sequential(
                self.init_(nn.Linear(self.hidden_size + self.num_program_tokens + self.hidden_size,
                                    self.hidden_size)),
                nn.Tanh(), self.init_(nn.Linear(self.hidden_size, self.num_program_tokens))
            ),
            device=self.device,
            dsl=dsl,
            max_program_length=max_program_length
        )
        
        state_encoder = nn.Sequential(
            self.init_(nn.Conv2d(self.state_shape[0], 32, 3, stride=1)), nn.ReLU(),
            self.init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), nn.Flatten(),
            self.init_(nn.Linear(32 * 4 * 4, self.hidden_size)), nn.ReLU()
        )
        
        # Input: a_i (A). Output: enc(a_i) (A).
        action_encoder = nn.Embedding(self.num_agent_actions, self.num_agent_actions)
        
        self.traj_rec = TrajectorySequenceDecoder(
            action_encoder,
            state_encoder,
            gru=nn.GRU(2 * self.hidden_size + self.num_agent_actions, self.hidden_size),
            mlp=nn.Sequential(
                self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
                self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
                self.init_(nn.Linear(self.hidden_size, self.num_agent_actions))
            ),
            device=self.device,
            max_demo_length=max_demo_length,
            num_agent_actions=self.num_agent_actions
        )
        
        # Encoder VAE utils
        self.quantizer = VectorQuantizer(self.device, vq_size, vq_dim, 0.25)
        self.vq_pooling = vq_pooling
        if self.vq_pooling or vq_dim != self.hidden_size:
            self.enc_linear = nn.Linear(self.hidden_size, vq_dim)
            self.dec_linear = nn.Linear(vq_dim, self.hidden_size)
        
        self.to(self.device)

    def sample_latent_vector(self, enc_hidden_state: torch.Tensor) -> torch.Tensor:
        if self.vq_pooling:
            return self.quantizer(self.enc_linear(enc_hidden_state)) #.detach() TODO ?
        else:
            return self.quantizer(enc_hidden_state)
    
    def get_latent_loss(self):
        return self.quantizer.get_loss()
    
    def forward(self, data_batch: tuple, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelReturn:
        s_h, a_h, a_h_masks, progs, progs_masks = data_batch
        
        enc_hidden_state = self.prog_encoder(progs, progs_masks)
        
        z = self.sample_latent_vector(enc_hidden_state)
        
        if self.vq_pooling:
            z = self.dec_linear(z)
        
        pred_progs, pred_progs_logits, pred_progs_masks = self.prog_decoder(
            z, progs, progs_masks, prog_teacher_enforcing
        )
        pred_a_h, pred_a_h_logits, pred_a_h_masks = self.traj_rec(
            z, s_h, a_h, a_h_masks, a_h_teacher_enforcing
        )
        
        return ModelReturn(z, pred_progs, pred_progs_logits, pred_progs_masks,
                           pred_a_h, pred_a_h_logits, pred_a_h_masks)
        
    def encode_program(self, prog: torch.Tensor):
        if prog.dim() == 1:
            prog = prog.unsqueeze(0)
        
        prog_mask = (prog != self.num_program_tokens - 1)
        
        enc_hidden_state = self.prog_encoder(prog, prog_mask)
        
        z = self.sample_latent_vector(enc_hidden_state)
        
        return z
    
    def decode_vector(self, z: torch.Tensor):
        pred_progs, _, pred_progs_masks = self.prog_decoder(z, None, None, False)
        
        pred_progs_tokens = []
        for prog, prog_mask in zip(pred_progs, pred_progs_masks):
            pred_progs_tokens.append([0] + prog[prog_mask].cpu().numpy().tolist())
        
        return pred_progs_tokens
    
    def get_losses_and_accs(self, data_batch: tuple, output: ModelReturn,
                                  loss_fn: nn.CrossEntropyLoss) -> tuple[Losses, Accuracies]:
        _, a_h, a_h_masks, progs, progs_masks = data_batch

        _, pred_progs, pred_progs_logits, pred_progs_masks,\
            pred_a_h, pred_a_h_logits, pred_a_h_masks = output
        
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
        
        if pred_a_h is not None:
            pred_a_h_logits = pred_a_h_logits.view(-1, pred_a_h_logits.shape[-1])
            pred_a_h_masks_flat = pred_a_h_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            a_h_masks_flat_combined = torch.max(a_h_masks_flat, pred_a_h_masks_flat).squeeze()
        
        if pred_progs is not None:
            pred_progs_logits = pred_progs_logits.view(-1, pred_progs_logits.shape[-1])
            pred_progs_masks_flat = pred_progs_masks.view(-1, 1)
            # We combine masks here to penalize predictions that are larger than ground truth
            progs_masks_flat_combined = torch.max(progs_masks_flat, pred_progs_masks_flat).squeeze()
        
        zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
        progs_loss, a_h_loss = zero_tensor, zero_tensor
        
        if pred_a_h is not None:
            a_h_loss = loss_fn(pred_a_h_logits[a_h_masks_flat_combined],
                               a_h_flat[a_h_masks_flat_combined].view(-1))
        
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
            if pred_a_h is not None:
                a_h_masks_combined = torch.max(a_h_masks, pred_a_h_masks)
                a_h_t_accuracy = (pred_a_h[a_h_masks_combined] == a_h[a_h_masks_combined]).float().mean()
                a_h_s_accuracy = (a_h == pred_a_h).min(dim=1).values.float().mean()
        
        return Losses(progs_loss, a_h_loss, latent_loss),\
            Accuracies(progs_t_accuracy, progs_s_accuracy, a_h_t_accuracy, a_h_s_accuracy)

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

                output = self(train_batch,
                              not disable_prog_teacher_enforcing,
                              not disable_a_h_teacher_enforcing)
                losses, accs = self.get_losses_and_accs(train_batch, output, loss_fn)
                
                total_loss = prog_loss_coef * losses.progs_rec +\
                    a_h_loss_coef * losses.a_h_rec +\
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
            
            if self.wandb_run is not None:
                self.wandb_run.log({
                    'epoch': epoch,
                    'train_total_loss': train_mean_total_loss,
                    **{'train_' + loss + '_loss': train_mean_losses[i] for i, loss in enumerate(Losses._fields)},
                    **{'train_' + acc + '_accuracy': train_mean_accs[i] for i, acc in enumerate(Accuracies._fields)}
                })
                
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
                    
                    total_loss = prog_loss_coef * losses.progs_rec +\
                        a_h_loss_coef * losses.a_h_rec +\
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
                    
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        'epoch': epoch,
                        'val_total_loss': val_mean_total_loss,
                        **{'val_' + loss + '_loss': val_mean_losses[i] for i, loss in enumerate(Losses._fields)},
                        **{'val_' + acc + '_accuracy': val_mean_accs[i] for i, acc in enumerate(Accuracies._fields)}
                    })
                    
                if val_mean_total_loss < best_val_return:
                    self.log(f'New best validation loss: {val_mean_total_loss}')
                    best_val_return = val_mean_total_loss
                    params_path = os.path.join(model_folder, f'best_val.ptp')
                    torch.save(self.state_dict(), params_path)
                    self.log(f'Parameters saved to {params_path}')
