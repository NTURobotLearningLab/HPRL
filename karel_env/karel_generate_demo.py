from __future__ import print_function, division, absolute_import, unicode_literals
import os
import importlib.util
import time
import pickle
import shutil
import pdb
import torch
import gym
import logging
import numpy as np
import random
import sys
import argparse
import errno
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.utils.rnn as rnn
from tensorboardX import SummaryWriter

sys.path.insert(0, '.')
from pretrain import customargparse
# from pretrain.CEM_analysis_dataset_programs import CEMModel_analysis
from pretrain.misc_utils import log_record_dict, create_directory
from fetch_mapping import fetch_mapping
from rl.envs import make_vec_envs
from rl import utils

from karel_env.dsl import get_DSL_option_v2
from karel_env import karel_option as karel
from karel_env.generator_option import KarelStateGenerator

from pygifsicle import optimize
import imageio
import json

def obs2action(obs_list):
    last_obs = obs_list[0]
    perception_dict = dict()
    perception_dict = {'frontIsClear':'', 'leftIsClear':'', 'rightIsClear':'', 'markersPresent':'', 'noMarkersPresent':''}
    action_arr = np.full((len(obs_list)-1,), -1, dtype=int)
    perception_arr = np.zeros((len(obs_list), 5), dtype=bool)
    action_list = ['' for _ in range(len(obs_list))]
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    for l in range(1, len(obs_list)+1):
        if l < len(obs_list):
            current_obs = obs_list[l]
        # print(last_obs[:, :, 6])
        # print(current_obs[:, :, 6])
        for i in range(1, config['input_height'] - 1):
            for j in range(1, config['input_width'] - 1):
                for k in range(4):
                    if last_obs[i][j][k]:
                        if last_obs[i+dx[k]][j+dy[k]][4]:
                            # perception_dict['frontIsClear'].append('False')
                            perception_dict['frontIsClear'] += '0'
                        else:
                            # perception_dict['frontIsClear'].append('True')
                            perception_dict['frontIsClear'] += '1'
                            perception_arr[l-1][0] = 1
                        if last_obs[i+dx[(k+1)%4]][j+dy[(k+1)%4]][4]:
                            # perception_dict['rightIsClear'].append('False')
                            perception_dict['rightIsClear'] += '0'
                        else:
                            # perception_dict['rightIsClear'].append('True')
                            perception_dict['rightIsClear'] += '1'
                            perception_arr[l-1][2] = 1
                        if last_obs[i+dx[(k+3)%4]][j+dy[(k+3)%4]][4]:
                            # perception_dict['leftIsClear'].append('False')
                            perception_dict['leftIsClear'] += '0'
                        else:
                            # perception_dict['leftIsClear'].append('True')
                            perception_dict['leftIsClear'] += '1'
                            perception_arr[l-1][1] = 1
                        if last_obs[i][j][6] or last_obs[i][j][7]:
                            # perception_dict['markerPresent'].append('False')
                            perception_dict['markersPresent'] += '1'
                            perception_dict['noMarkersPresent'] += '0'
                            perception_arr[l-1][3] = 1
                        else:
                            # perception_dict['markerPresent'].append('True')
                            perception_dict['markersPresent'] += '0'
                            perception_dict['noMarkersPresent'] += '1'
                            perception_arr[l-1][4] = 1
                        
                        if l < len(obs_list):
                            if all(x == 0 for x in current_obs[i][j][:4]):
                                action_list[l-1] = 'move'
                                action_arr[l-1] = 0
                            else:
                                if current_obs[i][j][(k+1)%4]:
                                    action_list[l-1] = 'turnRight'
                                    action_arr[l-1] = 2
                                elif current_obs[i][j][(k+3)%4]:
                                    action_list[l-1] = 'turnLeft'
                                    action_arr[l-1] = 1
                                elif current_obs[i][j][(k+2)%4]:
                                    action_list[l-1] = 'move'
                                    action_arr[l-1] = 0

                if l < len(obs_list) and action_arr[l-1] == -1:
                    if (last_obs[i][j][6] and not current_obs[i][j][6] and not current_obs[i][j][7]) or (last_obs[i][j][7] and not current_obs[i][j][7]):
                            action_list[l-1] = 'pickMarker'
                            action_arr[l-1] = 4
                    if (not last_obs[i][j][6] and not last_obs[i][j][7] and current_obs[i][j][6]) or (not last_obs[i][j][7] and current_obs[i][j][7]):
                            action_list[l-1] = 'putMarker'
                            action_arr[l-1] = 3
        
        if l < len(obs_list):
            last_obs = current_obs
            # if action_arr[l-1] == -1:
            #     print(l)
            #     raise RuntimeError(f"Action {l} is wrong.")
    action_list[-1] = 'end'

    print(len(obs_list))
    print(len(action_list))
    print(len(perception_dict['leftIsClear']))

    return action_list.copy(), perception_dict.copy(), action_arr.copy(), perception_arr.copy()


def run(config, logger):

    if config['logging']['wandb']:
        import wandb
        wandb.init(project="prl-nips", sync_tensorboard=True, name=config['outdir'].split('/')[-1])
    else:
        os.environ['WANDB_MODE'] = 'dryrun'

    # begin block: this block sets the device from the config
    if config['device'].startswith('cuda') and torch.cuda.is_available():
        device = torch.device(config['device'])
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        logger.warning('{} GPU not available, running on CPU'.format(__name__))

    # setup tensorboardX: create a summary writer
    writer = SummaryWriter(logdir=config['outdir'])

    # this line logs the device info
    logger.debug('{} Using device: {}'.format(__name__, device))

    # end block: this block looks good

    # begin block: this block sets random seed for the all the modules
    if config['seed'] is not None:
        logger.debug('{} Setting random seed'.format(__name__))
        seed = config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    if config['device'].startswith('cuda') and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # end block: this block looks good. if we have specified a seed, then we set it

    global_logs = {'info': {}, 'result': {}}

    # make dummy env to get action and observation space based on the environment
    custom_kwargs = {"config": config['args']}
    custom = True if "karel" or "CartPoleDiscrete" in config['env_name'] else False
    logger.debug('Using environment: {}'.format(config['env_name']))
    envs = make_vec_envs(config['env_name'], config['seed'], 1,
                         config['gamma'], os.path.join(config['outdir'], 'openai'), device, False, custom_env=custom,
                         custom_kwargs=custom_kwargs)

    # call the training function using the dataloader and the model
    dsl = get_DSL_option_v2(seed=seed, environment=config['rl']['envs']['executable']['name'])
    config['dsl']['num_agent_actions'] = len(dsl.action_functions) + 1      # +1 for a no-op action, just for filling
    # if config['algorithm'] == 'CEM':
    #     model = CEMModel_analysis(device, config, envs, dsl, logger, writer, global_logs, config['verbose'])
    # else:
    #     assert False, "algorith != CEM"
    
    # Add wandb logger to the model
    if config['logging']['wandb']:
        wandb.config.update(config)
    num_outputs = int(envs.action_space.high[0]) if not config['two_head'] else int(envs.action_space.high[0] - 1)
    num_program_tokens = num_outputs if not config['two_head'] else num_outputs + 1
    print("num_program_tokens: ", num_program_tokens)
    
    karel_world = karel.Karel_world(make_error=False)
    
    def save_gif(path, s_h, last_state_image_path=None):
        # create video
        frames = []
        for s in s_h:
            frames.append(np.uint8(karel_world.state2image(s=s).squeeze()))
        frames = np.stack(frames, axis=0)
        imageio.mimsave(path, frames, format='GIF-PIL', fps=5)
        # save last state as png file
        if last_state_image_path:
            imageio.imsave(last_state_image_path, frames[-1])

        optimize(path)

        return
    
    rewards = float(0)
    obs = envs.reset()
    for i in range(config['num_demo']):
        exec_dict = dict()
        for j in range(config['max_episode_steps']):
            program_str = open(config['program_file']).readlines()[j].strip()
            if i == 0:
                logger.debug('Input program {}: {}'.format(j, program_str))
            program_tokens = torch.from_numpy(np.array(dsl.str2intseq(program_str)[1:], dtype=np.int8))
            action = torch.unsqueeze(program_tokens, 0).repeat(1, 1).to(device)

            obs, reward, done, infos = envs.step(action)
            
            if 'episode' in infos[0].keys():
                # logger.debug('Sample {}_step{} : {}'.format(i, j, infos[0]['episode']['r']))
            # if j == config['num_step'] - 1:
                logger.debug('Sample {} : {}'.format(i, infos[0]['episode']['r']))
                rewards += float(infos[0]['episode']['r'])

                exec_dict['s_h'] = np.array(infos[0]['exec_data']['s_image_h_list'])
                exec_dict['s_h_len'] = exec_dict['s_h'].shape[0]
                _, _, _, exec_dict['perception_arr'] = obs2action(exec_dict['s_h'])
                exec_dict['action_arr'] = np.array(infos[0]['exec_data']['a_image_h_list'])

                print(exec_dict)
                print(exec_dict['s_h'].shape)
                print(exec_dict['action_arr'].shape)
                print(exec_dict['perception_arr'].shape)
                with open(os.path.join(config['outdir'], "{}_sample{}.json".format(config['env_task'], i)), "w") as outfile:
                    exec_dict['s_h'] = str(exec_dict['s_h'])
                    exec_dict['action_arr'] = str(exec_dict['action_arr'])
                    exec_dict['perception_arr'] = str(exec_dict['perception_arr'])
                    json.dump(exec_dict, outfile)
                
                save_video_path = os.path.join(config['outdir'], "{}_sample{}.gif".format(config['env_task'], i))
                save_last_state_image_path = os.path.join(config['outdir'], "{}_sample{}.png".format(config['env_task'], i))
                save_gif(save_video_path, infos[0]['exec_data']['s_image_h_list'], save_last_state_image_path)

    logger.debug('Average : {}'.format(rewards/config['num_demo']))  
        # for i, info in enumerate(infos):
            # print('Sample {} : {}'.format(i, info['exec_data']['mean_reward']))
        # print("Env Task: ", config['env_task'], " max reward: ", max_reward, " max_reward_program: ", max_reward_program)
        

    return


def _temp(config, args):

    args.task_file = config['rl']['envs']['executable']['task_file']
    args.grammar = config['dsl']['grammar']
    args.use_simplified_dsl = config['dsl']['use_simplified_dsl']
    args.task_definition = config['rl']['envs']['executable']['task_definition']
    args.execution_guided = config['rl']['policy']['execution_guided']



if __name__ == "__main__":
    
    torch.set_num_threads(1)

    t_init = time.time()
    parser = customargparse.CustomArgumentParser(description='syntax learner')

    # Add arguments (including a --configfile)
    parser.add_argument('-o', '--outdir',
                        help='Output directory for results', default='karel_demo')
    parser.add_argument('-c', '--configfile',
                        help='Input file for parameters, constants and initial settings')
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')
    parser.add_argument('--num_demo', type=int,
                        help='Demo number', default=10)
    parser.add_argument('--program_file',
                        help='File path/name of the program', required=True)


    # Parse arguments
    
    args = parser.parse_args()

    # FIXME: This is only for backwards compatibility to old parser, should be removed once we change the original
    # args.outdir = os.path.join(args.outdir, '%s-%s-%s-%s' % (args.prefix, args.grammar, args.seed, time.strftime("%Y%m%d-%H%M%S")))
    args.outdir = os.path.join(args.outdir, '%s-%s' % (args.env_task, args.seed))
    log_dir = os.path.expanduser(os.path.join(args.outdir, 'openai'))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    # fetch the mapping from prl tokens to dsl tokens
    if args.mapping_file is not None:
        args.dsl2prl_mapping, args.prl2dsl_mapping, args.dsl_tokens, args.prl_tokens = \
            fetch_mapping(args.mapping_file)
        args.use_simplified_dsl = True
        args.use_shorter_if = True if 'shorter_if' in args.mapping_file else False
    else:
        _, _, args.dsl_tokens, _ = fetch_mapping('mapping_karel2prl_new_vae_v2.txt')
        args.use_simplified_dsl = False

    config = customargparse.args_to_dict(args)
    config['args'] = args
    
    _temp(config, args)
    
    # TODO: shift this logic somewhere else
    # encode reward along with state and action if task defined by custom reward
    config['rl']['envs']['executable']['dense_execution_reward'] = config['rl']['envs']['executable'][
                                                                       'task_definition'] == 'custom_reward'

    # Create output directory if it does not already exist
    create_directory(config['outdir'])

    # Set up logger
    log_file = os.path.join(config['outdir'], config['logging']['log_file'])
    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')]
    logging.basicConfig(handlers=log_handlers, format=config['logging']['fmt'], level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print(config['logging'])
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    logger.disabled = (not config['verbose'])

    # Call the main method
    run_results = run(config, logger)

    # Final time
    t_final = time.time()
    logger.debug('{} Program finished in {} secs.'.format(__name__, t_final - t_init))
    print('{} Program finished in {} secs.'.format(__name__, t_final - t_init))
