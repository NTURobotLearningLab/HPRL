import copy
import glob
import os
import time
from collections import deque
import sys
import imageio
from pygifsicle import optimize

sys.path.insert(0, 'karel_env')

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl import utils
from rl.envs import make_vec_envs

from ppo_iko import algo
from ppo_iko.model_ppo_option import Policy
from ppo_iko.storage_option import RolloutStorage

from pretrain.models_option_new_vae import ProgramVAE
from karel_env import karel_option as karel


class PPOModel(object):
    def __init__(self, device, config, dummy_envs, dsl, logger, writer, global_logs, verbose):
        self.device = device
        self.config = config
        self.global_logs = global_logs
        self.verbose = verbose
        self.logger = logger
        self.writer = writer
        self.envs = dummy_envs
        self.dsl = dsl
        self.num_env_steps = config['PPO']['num_env_steps']
        self.num_steps = config['PPO']['num_steps']
        self.num_processes = config['PPO']['num_processes']
        self.use_linear_lr_decay = config['PPO']['use_linear_lr_decay']
        self.decoder_deterministic = config['PPO']['decoder_deterministic']
        self.algo = config['PPO']['algo']
        self.use_gae = config['PPO']['use_gae']
        self.gae_lambda = config['PPO']['gae_lambda']
        self.gamma = config['PPO']['gamma']
        self.use_proper_time_limits = config['PPO']['use_proper_time_limits']
        self.env_name = config['env_name']
        self.log_interval = config['log_interval']
        self.log_video_interval = config['log_video_interval']
        self.save_interval = config['save_interval']
        
        # create karel world for video image generation
        self._world = karel.Karel_world(make_error=False)

        # create log dir
        log_dir = os.path.expanduser(os.path.join(config['outdir'], 'PPO', 'openai_PPO'))
        log_dir_best = os.path.expanduser(os.path.join(config['outdir'], 'PPO', 'openai_PPO','best'))
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(log_dir_best)
        utils.cleanup_log_dir(eval_log_dir)

        # create env
        cfg_rl = config['rl']
        cfg_envs = config['rl']['envs']
        custom = True if "karel" or "CartPoleDiscrete" in cfg_envs['executable']['name'] else False
        logger.info('Using environment: {}'.format(cfg_envs['executable']['name']))
        
        assert config['mdp_type'] in [
                'ProgramEnv_option_new_vae', 
                'ProgramEnv_option_new_vae_v2',
                'ProgramEnv_option_new_vae_v2_reconstruct', 
                'ProgramEnv_option_new_vae_v2_key2door', 
                'ProgramEnv_option_new_vae_v2_key2door_fixed',
                'ProgramEnv_option_new_vae_v2_key2door_fixed_sparse',
                ]
        self.envs = make_vec_envs(
                cfg_envs['executable']['name'], 
                config['seed'], 
                config['PPO']['num_processes'], 
                cfg_rl['gamma'], 
                os.path.join(config['outdir'], 'PPO', 'openai_PPO'),
                device, 
                False, 
                custom_env=custom, 
                custom_env_type='program', 
                custom_kwargs={'config': config['args']})

        # create policy network
        base_kwargs = {
                'recurrent': config['PPO']['recurrent_policy'],
                'hidden_size': config['PPO']['hidden_size'],
                'input_height': config['input_height'],
                'input_width': config['input_width'],
                }
        self.actor_critic = Policy(self.envs, base_kwargs=base_kwargs)
        self.actor_critic.to(device)

        # create algo
        if self.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic,
                config['A2C']['value_loss_coef'],
                config['A2C']['entropy_coef'],
                lr=config['A2C']['lr'],
                eps=config['A2C']['eps'],
                alpha=config['A2C']['alpha'],
                max_grad_norm=config['A2C']['max_grad_norm'])
        elif self.algo == 'ppo':
            self.agent = algo.PPO_optionModel(
                self.actor_critic,
                config['PPO']['clip_param'],
                config['PPO']['ppo_epoch'],
                config['PPO']['num_mini_batch'],
                config['PPO']['value_loss_coef'],
                config['PPO']['entropy_coef'],
                lr=config['PPO']['lr'],
                eps=config['PPO']['eps'],
                max_grad_norm=config['PPO']['max_grad_norm'])
        else:
            raise NotImplementedError

        # create rollout buffer
        self.rollouts = RolloutStorage(
                self.num_steps,
                self.num_processes,
                self.envs.observation_space.shape, 
                config['PPO']['hidden_size'],
                self.actor_critic.recurrent_hidden_state_size)

        # load vae from checkpoint
        self.program_vae = ProgramVAE(self.envs, **config)
        self.program_vae.to(device)
        checkpt = self.config['net']['saved_params_path']
        assert checkpt is not None
        print('Loading params from {}'.format(checkpt))
        params = torch.load(checkpt, map_location=device)
        self.program_vae.load_state_dict(params[0], strict=True)
        self.tanh = self.program_vae.vae.tanh
        self.program_vae.eval() # TODO: Check if needed (for dropout layer)

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(device)

        self.episode_rewards        = deque(maxlen=10)
        self.episode_programs       = deque(maxlen=10)
        self.episode_len            = deque(maxlen=10)
        self.episode_primitive_len  = deque(maxlen=10)
        self.episode_s_h            = deque(maxlen=10)
    
    def save_gif(self, path, s_h):
        # create video
        frames = []
        for s in s_h:
            frames.append(np.uint8(self._world.state2image(s=s).squeeze()))
        frames = np.stack(frames, axis=0)
        imageio.mimsave(path, frames, format='GIF-PIL', fps=5)
        #optimize(path)

        return


    def train(self):
        start = time.time()
        num_updates = int(
            self.num_env_steps) // self.num_steps // self.num_processes
        for j in range(num_updates):
            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    self.agent.optimizer, j, num_updates,
                    self.agent.optimizer.lr if self.algo == "acktr" else self.lr)
    
            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, z, z_log_prob, recurrent_hidden_states = self.actor_critic.act(self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step], self.rollouts.masks[step])
    
                # use decoder for option program synthesis
                # TODO: Think on tanh latter
                with torch.no_grad():
                    tanh_z = self.tanh(z)
                    decoder_output = self.program_vae.vae.decoder(None, tanh_z, teacher_enforcing=False, deterministic=self.decoder_deterministic, evaluate=False)
                
                _, pred_programs, pred_programs_len, pred_programs_log_probs, pred_programs_output_logits, eop_pred_programs, eop_output_logits, pred_program_masks, pred_programs_dist_entropy = decoder_output
    
                # Obser reward and next obs
                obs, reward, done, infos = self.envs.step(pred_programs)
    
                for info in infos:
                    if 'episode' in info.keys():
                        # TODO: check on how info['episode']['r'] is added
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_programs.append(info['exec_data']['program_str_history'])
                        self.episode_len.append(info['exec_data']['program_step'])
                        self.episode_primitive_len.append(info['exec_data']['primitive_episode_len'])
                        self.episode_s_h.append(info['exec_data']['s_image_h_list'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                # TODO: check this usage
                bad_masks = torch.FloatTensor([[1.0] for i in range(len(infos))])
                #bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                self.rollouts.insert(obs, recurrent_hidden_states, z, z_log_prob, value, reward, masks, bad_masks)
    
            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()
    
            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.gae_lambda, self.use_proper_time_limits)
    
            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
    
            self.rollouts.after_update()
    
            # save for every interval-th episode or for the last epoch
            if (j % self.save_interval == 0 or j == num_updates - 1):
                save_path = os.path.join(self.config["outdir"], 'PPO')
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass
    
                torch.save(self.actor_critic, 
                        os.path.join(save_path, self.env_name + "_" + self.config['env_task'] + "_model.pt"))
                print("Saving model to path: ", os.path.join(save_path, self.env_name + "_" + self.config['env_task'] + "_model.pt"))

            if j % self.log_video_interval == 0 and j > 0:
                print("Log Videos...")
                video_len = [len(s_h) for s_h in self.episode_s_h]
                print("video len mean/min/max {}/{}/{}:", np.mean(video_len), np.min(video_len), np.max(video_len))
                # create video log 
                log_video_dir = os.path.expanduser(os.path.join(self.config['outdir'], 'PPO', 'video_{}_update{}'.format(self.config['env_task'], j)))
                utils.cleanup_log_dir(log_video_dir)
                save_video_dir_path = os.path.join(self.config['outdir'], 'PPO', 'video_{}_update{}'.format(self.config['env_task'], j))
                for idx, s_h in enumerate(self.episode_s_h):
                    save_video_path = os.path.join(save_video_dir_path, "{}_update{}_sample{}.gif".format(self.config['env_task'], j, idx)),
                    self.save_gif(save_video_path[0], s_h)
                for s_idx, (ri, pi) in enumerate(zip(self.episode_rewards, self.episode_programs)):
                    self.writer.add_text(
                            'program/Num_update{}_Sample{}'.format(j, s_idx), 
                            'reward_env: {} program: {} '.format(ri, pi), 
                            total_num_steps
                            )
   
            if j % self.log_interval == 0 and len(self.episode_rewards) > 1:
                total_num_steps = (j + 1) * self.num_processes * self.num_steps
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {}".format(j, total_num_steps, int(total_num_steps / (end - start))), end=' ')
                print("Last {} training episodes: mean/min/max reward {:.4f}/{:.4f}/{:.4f}".format(len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards)), end=' ')
                print("Last {} episode len mean/median {:.1f}/{:.1f}".format(len(self.episode_len), np.mean(self.episode_len), np.median(self.episode_len)), end=' ')
                print("Last {} primitive len mean/median {:.1f}/{:.1f}".format(len(self.episode_primitive_len), np.mean(self.episode_primitive_len), np.median(self.episode_primitive_len)))
            # Add logs to TB
            self.writer.add_scalar('train/FPS', int(total_num_steps / (end - start)), total_num_steps)
            self.writer.add_scalar('train/mean_episode_reward', np.mean(self.episode_rewards), total_num_steps)
            self.writer.add_scalar('train/max_episode_reward', np.max(self.episode_rewards), total_num_steps)
            self.writer.add_scalar('train/min_episode_reward', np.min(self.episode_rewards), total_num_steps)
            self.writer.add_scalar('train/mean_episode_len', np.mean(self.episode_len), total_num_steps)
            self.writer.add_scalar('train/mean_episode_primitive_len', np.mean(self.episode_primitive_len), total_num_steps)
            self.writer.add_scalar('train/dist_entropy', dist_entropy, total_num_steps)
            self.writer.add_scalar('train/value_loss', value_loss, total_num_steps)
            self.writer.add_scalar('train/action_loss', action_loss, total_num_steps)
   
