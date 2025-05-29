import os
import time
import torch
import pickle
import random
import logging
from torch.utils.data import DataLoader
import wandb
from tensorboardX import SummaryWriter
import sys
sys.path.append('.')

from pretrain.SupervisedModel_option_new_vae import SupervisedModel
from pretrain.misc_utils import create_directory
from pretrain.customargparse import CustomArgumentParser, args_to_dict
from fetch_mapping import fetch_mapping
from karel_env.dsl import get_DSL_option_v2
from pretrain.trainer_option_new_vae_L30 import make_datasets  # must extract from original script if not yet
from rl.envs import make_vec_envs
from rl import utils

def _temp(config, args):

    args.task_file = config['rl']['envs']['executable']['task_file']
    args.grammar = config['dsl']['grammar']
    args.use_simplified_dsl = config['dsl']['use_simplified_dsl']
    args.task_definition = config['rl']['envs']['executable']['task_definition']
    args.execution_guided = config['rl']['policy']['execution_guided']

def run(config, logger):
    if config['device'].startswith('cuda') and torch.cuda.is_available():
        device = torch.device(config['device'])
    else:
        device = torch.device('cpu')
        logger.warning('{} GPU not available, running on CPU'.format(__name__))

    writer = SummaryWriter(logdir=config['outdir'])

    if config['seed'] is not None:
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # DSL
    dsl = get_DSL_option_v2(seed=config['seed'], environment=config['rl']['envs']['executable']['name'])
    config['dsl']['num_agent_actions'] = len(dsl.action_functions) + 1

    # Env
    custom_kwargs = {"config": config['args']}
    custom = True if "karel" or "CartPoleDiscrete" in config['env_name'] else False
    envs = make_vec_envs(config['env_name'], config['seed'], 1,
                         config['gamma'], "/home/hubertchang/HPRL/pretrain/output_dir_new_vae_L40_1m_30epoch_20230104/LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear_cuda8-handwritten-123-20250508-155204/openai", device, False,
                         custom_env=custom, custom_kwargs=custom_kwargs)

    # Model
    model = SupervisedModel(device, config, envs, dsl, logger, writer, {}, config['verbose'])

    # Load checkpoint
    # ckpt_path = os.path.join(config['outdir'], 'best_valid_params.ptp')
    # print("Loading checkpoint:", ckpt_path)
    ckpt_path = "/home/vincentchang/HPRL/pretrain/action_encoder_norope_alldata/LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear_cuda8-handwritten-123-20250510-022930/best_valid_params.ptp"
    model.load_net(ckpt_path)

    # Datasets
    train_dataset, _, _ = make_datasets(config['datadir'], config,
                                        model.num_program_tokens,
                                        config['dsl']['num_agent_actions'],
                                        device, logger, dsl)

    train_loader = DataLoader(train_dataset,
                              batch_size=config['train']['batch_size'],
                              shuffle=False,
                              **config['data_loader'])

    # Evaluate
    print("Evaluating on training dataset...")
    model.evaluate(train_loader)
    print("Done.")

if __name__ == "__main__":
    run_name = input("Enter run name: ")
    
    from cfg_option_new_vae import config  # your config file
    wandb.init(project="DRLGS_eval", name = run_name, config=config)
    parser = CustomArgumentParser(description='Run supervised model on training data')
    parser.add_argument('-o', '--outdir',
                        help='Output directory for results', default='pretrain/output_dir')
    parser.add_argument('-d', '--datadir',
                        help='dataset directory containing data.hdf5 and id.txt')
    parser.add_argument('-dc', '--datadirCheck',
                        help='dataset directory containing checked (valid) id.txt')
    parser.add_argument('-c', '--configfile',
                        help='Input file for parameters, constants and initial settings')
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')
    parser.add_argument('--linear_inter_units',
                        help='Intermediate unit numbers of linear layers', type=int, nargs='+')

    # Parse arguments
    args = parser.parse_args()

    # DSL mappings
    if args.mapping_file is not None:
        args.dsl2prl_mapping, args.prl2dsl_mapping, args.dsl_tokens, args.prl_tokens = \
            fetch_mapping(args.mapping_file)
        args.use_simplified_dsl = True
        args.use_shorter_if = True if 'shorter_if' in args.mapping_file else False
    else:
        _, _, args.dsl_tokens, _ = fetch_mapping('mapping_karel2prl_new_vae_v2.txt')
        args.use_simplified_dsl = False

    # Compose config dictionary from args
    config = args_to_dict(args)
    _temp(config, args)
    config['args'] = args

    # Fill in task-related fields
    config['rl']['envs']['executable']['dense_execution_reward'] = \
        config['rl']['envs']['executable']['task_definition'] == 'custom_reward'

    # Setup output
    create_directory(config['outdir'])

    # Logger
    log_file = os.path.join(config['outdir'], config['logging']['log_file'])
    log_handlers = [logging.StreamHandler(), logging.FileHandler(log_file, mode='w')]
    logging.basicConfig(handlers=log_handlers, format=config['logging']['fmt'], level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    logger.disabled = not config['verbose']

    # Run
    run(config, logger)
