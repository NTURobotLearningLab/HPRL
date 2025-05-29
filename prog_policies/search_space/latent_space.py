from __future__ import annotations
import torch
from torch.autograd import Variable

from ..base.dsl import dsl_nodes, BaseDSL
from leaps.pretrain.models import ProgramVAE
from leaps.rl.envs import make_vec_envs
from leaps.karel_env.dsl import get_DSL
from leaps.pretrain.customargparse import CustomArgumentParser, args_to_dict
from leaps.fetch_mapping import fetch_mapping

from .base_space import BaseSearchSpace

class LatentSpace(BaseSearchSpace):
    
    def __init__(self, dsl: BaseDSL, sigma: float = 0.25) -> None:
        super().__init__(dsl, sigma)
        # Procedure to load LEAPS model using the authors' provided config
        torch.set_num_threads(1)
        parser = CustomArgumentParser()
        parser.add_argument('-c', '--configfile')
        parser.set_defaults(configfile='leaps/pretrain/cfg.py')
        args, _ = parser.parse_known_args()
        _, _, args.dsl_tokens, _ = fetch_mapping('leaps/mapping_karel2prl.txt')
        args.use_simplified_dsl = False
        args.device = 'cpu'
        args.num_lstm_cell_units = 256
        config = args_to_dict(args)
        args.task_file = config['rl']['envs']['executable']['task_file']
        args.grammar = config['dsl']['grammar']
        args.use_simplified_dsl = config['dsl']['use_simplified_dsl']
        args.task_definition = config['rl']['envs']['executable']['task_definition']
        args.execution_guided = config['rl']['policy']['execution_guided']
        config['args'] = args
        seed = self.torch_rng.get_state().cpu().numpy()[0]
        torch.manual_seed(seed)
        envs = make_vec_envs(config['env_name'], seed, 1,
                            config['gamma'], None, self.torch_device, False,
                            custom_env=True, custom_kwargs={'config': config['args']})
        self.leaps_dsl = get_DSL(seed=seed)
        config['dsl']['num_agent_actions'] = len(self.leaps_dsl.action_functions) + 1
        self.latent_model = ProgramVAE(envs, **config)
        params = torch.load('leaps/weights/LEAPS/best_valid_params.ptp', map_location=self.torch_device)
        self.latent_model.load_state_dict(params[0], strict=False)
        self.hidden_size = self.latent_model.recurrent_hidden_state_size
    
    def _decode(self, individual: torch.Tensor) -> dsl_nodes.Program:
        """Decodes a single latent vector into a program using LEAPS

        Args:
            individual (torch.Tensor): latent vector

        Returns:
            dsl_nodes.Program: decoded program
        """
        population = individual.unsqueeze(0)
        _, progs, progs_len, _, _, _, _, _, _ = self.latent_model.vae.decoder(
            None, population, teacher_enforcing=False, deterministic=True, evaluate=False
        )
        prog = progs.numpy().tolist()[0]
        prog_len = progs_len.numpy().tolist()[0][0]
        # Model outputs tokens starting from index 1
        prog_str = self.leaps_dsl.intseq2str([0] + prog[:prog_len])
        prog = self.dsl.parse_str_to_node(prog_str)
        return prog
    
    def _encode(self, program: dsl_nodes.Program) -> torch.Tensor:
        """Encodes a program into a latent vector using LEAPS

        Args:
            program (dsl_nodes.Program): program

        Returns:
            torch.Tensor: latent vector
        """
        init_prog_str = self.dsl.parse_node_to_str(program)
        init_prog_intseq = self.leaps_dsl.str2intseq(init_prog_str)
        init_prog_len = torch.tensor([len(init_prog_intseq)])
        init_prog_intseq += [self.dsl.t2i['<pad>']] * (45 - init_prog_len)
        init_prog_torch = torch.tensor(init_prog_intseq, dtype=torch.long, device=self.torch_device).unsqueeze(0)
        _, init_enc_hxs = self.latent_model.vae.encoder(init_prog_torch, init_prog_len)
        init_latent = self._sample_latent(init_enc_hxs.squeeze(0))
        init_latent = init_latent.squeeze(0)
        return init_latent
    
    def _sample_latent(self, h_enc: torch.Tensor) -> torch.Tensor:
        mu = self.latent_model.vae._enc_mu(h_enc)
        log_sigma = self.latent_model.vae._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(self.np_rng.normal(0, 1, size=sigma.size())).to(torch.float).to(h_enc.device)
        return mu + sigma * Variable(std_z, requires_grad=False)
    
    def initialize_individual(self) -> tuple[torch.Tensor, dsl_nodes.Program]:
        """Initializes a tuple of latent vector and associated program from a normal distribution

        Returns:
            tuple[torch.Tensor, dsl_nodes.Program]: latent vector and associated program
        """
        while True:
            try:
                latent = torch.randn(self.hidden_size, generator=self.torch_rng, device=self.torch_device)
                prog = self._decode(latent) # Check if it's a valid program
                break
            except (AssertionError, IndexError): # In case of invalid program, try again
                continue
        return latent, prog
    
    def get_neighbors(self, individual: torch.Tensor, k: int = 1) -> list[tuple[torch.Tensor, dsl_nodes.Program]]:
        """Returns k neighbors of a given latent vector

        Args:
            individual (torch.Tensor): Latent vector
            k (int, optional): Number of neighbors. Defaults to 1.

        Raises:
            Exception: If no valid neighbor is found after 50 tries

        Returns:
            list[tuple[torch.Tensor, dsl_nodes.Program]]: List of individuals as tuples of
            latent vector and associated program
        """
        neighbors = []
        for _ in range(k):
            n_tries = 0
            while n_tries < 50:
                try:
                    neighbor = individual + self.sigma * torch.randn(
                        self.hidden_size, generator=self.torch_rng, device=self.torch_device
                    )
                    prog = self._decode(neighbor) # Check if it's a valid program
                    break
                except (AssertionError, IndexError): # In case of invalid program, try again
                    n_tries += 1
                    continue
            if n_tries >= 50: 
                neighbor = individual
                prog = self._decode(neighbor)
            neighbors.append((neighbor, prog))
        return neighbors
