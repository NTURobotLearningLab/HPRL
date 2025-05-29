from __future__ import annotations
from typing import NamedTuple
from logging import Logger
import numpy as np
import torch
from torch import nn
import wandb

from prog_policies.base import BaseDSL, BaseEnvironment

from ..utils import init, init_gru, EnvironmentBatch
from ..syntax_checker import SyntaxChecker


class ModelReturn(NamedTuple):
    z: torch.Tensor
    progs: torch.Tensor
    progs_logits: torch.Tensor
    progs_masks: torch.Tensor
    a_h: torch.Tensor
    a_h_logits: torch.Tensor
    a_h_masks: torch.Tensor


class ProgramSequenceEncoder(nn.Module):
    
    def __init__(self, token_encoder: nn.Module, gru: nn.GRU):
        super().__init__()
        self.token_encoder = token_encoder
        self.gru = gru
        init_gru(self.gru)
        
    def forward(self, progs: torch.Tensor, progs_mask: torch.Tensor):
        if len(progs.shape) == 3:
            batch_size, demos_per_program, _ = progs.shape
            progs = progs.view(batch_size * demos_per_program, -1)
            progs_mask = progs_mask.view(batch_size * demos_per_program, -1)
        
        progs_len = progs_mask.squeeze(-1).sum(dim=-1).cpu()
        
        enc_progs = self.token_encoder(progs)
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            enc_progs, progs_len, batch_first=True, enforce_sorted=False
        )
        
        _, enc_hidden_state = self.gru(packed_inputs)
        
        return enc_hidden_state.squeeze(0)


class ProgramSequenceDecoder(nn.Module):
    
    def __init__(self, token_encoder: nn.Module, gru: nn.GRU, mlp: nn.Module,
                 device: torch.device, dsl: BaseDSL, max_program_length = 45,
                 only_structure=False):
        super().__init__()
        self.device = device
        self.token_encoder = token_encoder
        self.gru = gru
        self.mlp = mlp
        self.softmax = nn.LogSoftmax(dim=-1)
        self.num_program_tokens = len(dsl.get_tokens())
        self.syntax_checker = SyntaxChecker(dsl, self.device, only_structure=only_structure)
        self.only_structure = only_structure
        self.max_program_length = max_program_length
        init_gru(self.gru)
        
    def get_syntax_mask(self, batch_size: int, current_tokens: torch.Tensor, grammar_state: list):
        out_of_syntax_list = []
        out_of_syntax_mask = torch.zeros((batch_size, self.num_program_tokens), dtype=torch.bool, device=self.device)

        for program_idx, inp_token in enumerate(current_tokens):
            inp_dsl_token = inp_token.detach().cpu().numpy().item()
            out_of_syntax_list.append(self.syntax_checker.get_sequence_mask(grammar_state[program_idx],
                                                                            [inp_dsl_token]).to(self.device))
        torch.cat(out_of_syntax_list, 0, out=out_of_syntax_mask)
        out_of_syntax_mask = out_of_syntax_mask.squeeze()
        syntax_mask = torch.where(out_of_syntax_mask,
                                  -torch.finfo(torch.float32).max * torch.ones_like(out_of_syntax_mask).float(),
                                  torch.zeros_like(out_of_syntax_mask).float())

        return syntax_mask, grammar_state
    
    def forward(self, latent: torch.Tensor, progs: torch.Tensor, progs_mask: torch.Tensor,
               prog_teacher_enforcing = True):
        if progs is not None:
            if len(progs.shape) == 3:
                b, demos_per_program, _ = progs.shape
                progs = progs.view(b * demos_per_program, -1)
                progs_mask = progs_mask.view(b * demos_per_program, -1)
        
        batch_size, _ = latent.shape
        
        gru_hidden_state = latent.unsqueeze(0)
        
        # Initialize tokens as DEFs
        current_tokens = torch.zeros((batch_size), dtype=torch.long, device=self.device)
        
        grammar_state = [self.syntax_checker.get_initial_checker_state()
                         for _ in range(batch_size)]
        
        pred_progs = []
        pred_progs_logits = []
        
        for i in range(1, self.max_program_length):
            token_embedding = self.token_encoder(current_tokens)
            gru_inputs = torch.cat((token_embedding, latent), dim=-1)
            gru_inputs = gru_inputs.unsqueeze(0)
            
            gru_output, gru_hidden_state = self.gru(gru_inputs, gru_hidden_state)
            
            mlp_input = torch.cat([gru_output.squeeze(0), token_embedding, latent], dim=1)
            pred_token_logits = self.mlp(mlp_input)
            
            syntax_mask, grammar_state = self.get_syntax_mask(batch_size, current_tokens, grammar_state)
            
            pred_token_logits += syntax_mask
            
            pred_tokens = self.softmax(pred_token_logits).argmax(dim=-1)
            
            pred_progs.append(pred_tokens)
            pred_progs_logits.append(pred_token_logits)
            
            if prog_teacher_enforcing:
                # Enforce next token with ground truth
                current_tokens = progs[:, i].view(batch_size)
            else:
                # Pass current prediction to next iteration
                current_tokens = pred_tokens.view(batch_size)
        
        pred_progs = torch.stack(pred_progs, dim=1)
        pred_progs_logits = torch.stack(pred_progs_logits, dim=1)
        pred_progs_masks = (pred_progs != self.num_program_tokens - 1)
        
        return pred_progs, pred_progs_logits, pred_progs_masks


class TrajectorySequenceDecoder(nn.Module):
    
    def __init__(self, action_encoder: nn.Module, state_encoder: nn.Module, gru: nn.GRU,
                 mlp: nn.Module, device: torch.device, max_demo_length = 100,
                 num_agent_actions = 5):
        super().__init__()
        self.device = device
        self.action_encoder = action_encoder
        self.state_encoder = state_encoder
        self.gru = gru
        self.mlp = mlp
        self.softmax = nn.LogSoftmax(dim=-1)
        self.max_demo_length = max_demo_length
        self.num_agent_actions = num_agent_actions
        init_gru(self.gru)
    
    def env_init(self, states: torch.Tensor):
        states_np = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        # states_np = np.moveaxis(states_np,[-1,-2,-3], [-2,-3,-1])
        self._envs = EnvironmentBatch(states_np)

    def env_step(self, actions: torch.Tensor):
        # states_np = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        # states_np = np.moveaxis(states_np,[-1,-2,-3], [-2,-3,-1])
        # assert states_np.shape[-1] == 16
        # karel world expects H x W x C
        new_states = self._envs.step(actions.detach().cpu().numpy())
        # new_states = np.moveaxis(new_states,[-1,-2,-3], [-3,-1,-2])
        new_states = torch.tensor(new_states, dtype=torch.float32, device=self.device)
        return new_states
    
    def forward(self, latent: torch.Tensor, s_h: torch.Tensor, a_h: torch.Tensor,
                a_h_mask: torch.Tensor, a_h_teacher_enforcing = True, perceptions=False):
        if not perceptions:
            batch_size, demos_per_program, _, c, h, w = s_h.shape
            # Taking only first state and squeezing over first 2 dimensions
            current_state = s_h[:, :, 0, :, :, :].view(batch_size*demos_per_program, c, h, w)
        else:
            batch_size, demos_per_program, _, perc_size = s_h.shape
            current_state = s_h[:, :, 0, :].view(batch_size*demos_per_program, perc_size)
        
        ones = torch.ones((batch_size*demos_per_program, 1), dtype=torch.long, device=self.device)
        current_action = (self.num_agent_actions - 1) * ones
        
        _, hidden_size = latent.shape
        
        z_repeated = latent.unsqueeze(1).repeat(1, demos_per_program, 1)
        z_repeated = z_repeated.view(batch_size*demos_per_program, hidden_size)
        
        gru_hidden = z_repeated.unsqueeze(0)
        
        pred_a_h = []
        pred_a_h_logits = []
        
        if not a_h_teacher_enforcing:
            self.env_init(current_state)
        
        terminated_policy = torch.zeros_like(current_action, dtype=torch.bool, device=self.device)
        
        mask_valid_actions = torch.tensor((self.num_agent_actions - 1) * [-torch.finfo(torch.float32).max]
                                          + [0.], device=self.device)
        
        for i in range(1, self.max_demo_length):
            enc_state = self.state_encoder(current_state)
            
            enc_action = self.action_encoder(current_action.squeeze(-1))
            
            gru_inputs = torch.cat((z_repeated, enc_state, enc_action), dim=-1)
            gru_inputs = gru_inputs.unsqueeze(0)
            
            gru_out, gru_hidden = self.gru(gru_inputs, gru_hidden)
            gru_out = gru_out.squeeze(0)
            
            pred_action_logits = self.mlp(gru_out)
            
            masked_action_logits = pred_action_logits + terminated_policy * mask_valid_actions
            
            current_action = self.softmax(masked_action_logits).argmax(dim=-1).view(-1, 1)
            
            pred_a_h.append(current_action)
            pred_a_h_logits.append(pred_action_logits)
            
            # Apply teacher enforcing while training
            if a_h_teacher_enforcing:
                if not perceptions:
                    current_state = s_h[:, :, i, :, :, :].view(batch_size*demos_per_program, c, h, w)
                else:
                    current_state = s_h[:, :, i, :].view(batch_size*demos_per_program, perc_size)
                current_action = a_h[:, :, i].view(batch_size*demos_per_program, 1)
            # Otherwise, step in actual environment to get next state
            else:
                current_state = self.env_step(current_action)
                
            terminated_policy = torch.logical_or(current_action == self.num_agent_actions - 1,
                                                 terminated_policy)
    
        pred_a_h = torch.stack(pred_a_h, dim=1).squeeze(-1)
        pred_a_h_logits = torch.stack(pred_a_h_logits, dim=1)
        pred_a_h_masks = (pred_a_h != self.num_agent_actions - 1)
        
        return pred_a_h, pred_a_h_logits, pred_a_h_masks


class BaseVAE(nn.Module):
    """Base class for all program VAEs. Implements general functions used by subclasses.
    Do not directly instantiate this class.
    """    
    def __init__(self, dsl: BaseDSL, device: torch.device, env_cls: type[BaseEnvironment],
                 env_args: dict, max_program_length = 45, max_demo_length = 100, model_seed = 1,
                 hidden_size = 256, model_params_path: str = None, logger: Logger = None,
                 name: str = None, wandb_args: dict = None):
        super().__init__()
        
        if name is None:
            name = self.__class__.__name__
        self.name = name
        
        torch.manual_seed(model_seed)

        self.env_cls = env_cls
        self.env_args = env_args

        self.device = device
        
        self.max_demo_length = max_demo_length
        self.max_program_length = max_program_length
        
        # Z
        self.hidden_size = hidden_size
        
        # A
        self.num_agent_actions = len(dsl.get_actions()) + 1 # +1 because we have a NOP action
        
        # T
        self.num_program_tokens = len(dsl.get_tokens()) # dsl includes <pad> and <HOLE> tokens
        
        self.pad_token = dsl.t2i['<pad>']
        
        self.init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0), nn.init.calculate_gain('relu'))
        
        # CxHxW
        env = self.env_cls(**self.env_args)
        self.state_shape = env.state_shape
        
        self.perceptions_size = len(env.bool_features_list)
        
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # syntax_checker_tokens = dsl.get_tokens()
        # syntax_checker_tokens.append('<pad>')
        # self.T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
        self.syntax_checker = SyntaxChecker(dsl, self.device)
        
        self.to(self.device)
        
        if model_params_path is not None:
            self.load_state_dict(torch.load(model_params_path, map_location=self.device))
            
        self.logger = logger
        
        if wandb_args:
            self.wandb_run = wandb.init(
                name=self.name,
                **wandb_args
            )
        else:
            self.wandb_run = None
    
    def log(self, message: str):
        if self.logger is not None:
            self.logger.info(f'[{self.__class__.__name__}] {message}')
        else:
            print(f'[{self.__class__.__name__}] {message}')

    def forward(self, data_batch: tuple, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelReturn:
        raise NotImplementedError
    
    def encode_program(self, prog: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def decode_vector(self, z: torch.Tensor) -> list[list[int]]:
        raise NotImplementedError
