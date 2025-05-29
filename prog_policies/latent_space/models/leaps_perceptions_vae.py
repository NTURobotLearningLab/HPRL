from __future__ import annotations
from logging import Logger
import torch
import torch.nn as nn

from prog_policies.base import BaseDSL, BaseEnvironment

from ..utils import init_gru
from .base_vae import ModelReturn
from .leaps_vae import LeapsVAE


class LeapsPerceptionsVAE(LeapsVAE):
    """Reproduction of program VAE used in LEAPS paper.
    """    
    def __init__(self, dsl: BaseDSL, device: torch.device, env_cls: type[BaseEnvironment],
                 env_args: dict, max_program_length = 45, max_demo_length = 100, model_seed = 1,
                 hidden_size = 256, logger: Logger = None, name: str = None):
        super().__init__(dsl, device, env_cls, env_args, max_program_length, max_demo_length,
                         model_seed, hidden_size, logger=logger, name=name)
        
        # Input: enc(perc_i) (Z), enc(a_i) (A), z (Z). Output: pol_out (Z).
        self.policy_gru = nn.GRU(
            self.hidden_size + self.perceptions_size + self.num_agent_actions, self.hidden_size
        )
        init_gru(self.policy_gru)
        
        # Input: p_i (P). Output: enc(s_i) (P).
        self.perc_encoder = nn.Sequential(
            self.init_(nn.Linear(self.perceptions_size, self.perceptions_size)), nn.ReLU()
        )
        
        self.to(self.device)

    def policy_executor(self, z: torch.Tensor, perc_h: torch.Tensor, a_h: torch.Tensor,
                        a_h_mask: torch.Tensor, a_h_teacher_enforcing = True):
        batch_size, demos_per_program, _, perc_size = perc_h.shape
        
        # Taking only first state and squeezing over first 2 dimensions
        current_perc = perc_h[:, :, 0, :].view(batch_size*demos_per_program, perc_size)
        
        ones = torch.ones((batch_size*demos_per_program, 1), dtype=torch.long, device=self.device)
        current_action = (self.num_agent_actions - 1) * ones
        
        z_repeated = z.unsqueeze(1).repeat(1, demos_per_program, 1)
        z_repeated = z_repeated.view(batch_size*demos_per_program, self.hidden_size)
        
        gru_hidden = z_repeated.unsqueeze(0)
        
        pred_a_h = []
        pred_a_h_logits = []
        
        if not a_h_teacher_enforcing:
            raise NotImplementedError("Policy executor only supports teacher enforcing")
        
        terminated_policy = torch.zeros_like(current_action, dtype=torch.bool, device=self.device)
        
        mask_valid_actions = torch.tensor((self.num_agent_actions - 1) * [-torch.finfo(torch.float32).max]
                                          + [0.], device=self.device)
        
        for i in range(1, self.max_demo_length):
            enc_perc = self.perc_encoder(current_perc)
            
            enc_action = self.action_encoder(current_action.squeeze(-1))
            
            gru_inputs = torch.cat((z_repeated, enc_perc, enc_action), dim=-1)
            gru_inputs = gru_inputs.unsqueeze(0)
            
            gru_out, gru_hidden = self.policy_gru(gru_inputs, gru_hidden)
            gru_out = gru_out.squeeze(0)
            
            pred_action_logits = self.policy_mlp(gru_out)
            
            masked_action_logits = pred_action_logits + terminated_policy * mask_valid_actions
            
            current_action = self.softmax(masked_action_logits).argmax(dim=-1).view(-1, 1)
            
            pred_a_h.append(current_action)
            pred_a_h_logits.append(pred_action_logits)
            
            # Apply teacher enforcing while training
            if a_h_teacher_enforcing:
                current_perc = perc_h[:, :, i, :].view(batch_size*demos_per_program, perc_size)
                current_action = a_h[:, :, i].view(batch_size*demos_per_program, 1)
            # Otherwise, step in actual environment to get next state
            else:
                raise NotImplementedError("Policy executor only supports teacher enforcing")
                
            terminated_policy = torch.logical_or(current_action == self.num_agent_actions - 1,
                                                 terminated_policy)
    
        pred_a_h = torch.stack(pred_a_h, dim=1).squeeze(-1)
        pred_a_h_logits = torch.stack(pred_a_h_logits, dim=1)
        pred_a_h_masks = (pred_a_h != self.num_agent_actions - 1)
        
        return pred_a_h, pred_a_h_logits, pred_a_h_masks
    
    def forward(self, data_batch: tuple, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelReturn:
        perc_h, a_h, a_h_masks, progs, progs_masks = data_batch
        
        z = self.encode(progs, progs_masks)
        
        pred_progs, pred_progs_logits, pred_progs_masks = self.decode(
            z, progs, progs_masks, prog_teacher_enforcing
        )
        pred_a_h, pred_a_h_logits, pred_a_h_masks = self.policy_executor(
            z, perc_h, a_h, a_h_masks, a_h_teacher_enforcing
        )
        
        return ModelReturn(z, pred_progs, pred_progs_logits, pred_progs_masks,
                           pred_a_h, pred_a_h_logits, pred_a_h_masks)
        
    