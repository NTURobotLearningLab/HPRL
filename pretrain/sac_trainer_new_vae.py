#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
from collections import deque
import pickle as pkl
import imageio
from pygifsicle import optimize

from rl.envs import make_vec_envs

from sac_iko.actor import DiagGaussianActor
from sac_iko.critic import DoubleQCritic 
#from sac_iko.critic_shareCNN import DoubleQCritic 
from sac_iko.sac import SACAgent
from sac_iko.replay_buffer import ReplayBuffer
from sac_iko import utils
from pretrain.models_option_new_vae import ProgramVAE
from karel_env import karel_option as karel


class SACModel(object):
    def __init__(self, device, cfg, dummy_envs, dsl, logger, writer, global_logs, verbose):
        self.device                 = device
        self.cfg                    = cfg
        self.global_logs            = global_logs
        self.verbose                = verbose
        self.logger                 = logger
        self.writer                 = writer
        self.envs                   = dummy_envs
        self.dsl                    = dsl
        self.env_name               = cfg['env_name']
        self.log_interval           = cfg['log_interval']
        self.log_video_interval     = cfg['log_video_interval']
        self.save_interval          = cfg['save_interval']
        self.log_histogram_interval = cfg['SAC']['agent']['log_histogram_interval']
        self.learnable_temperature  = cfg['SAC']['agent']['learnable_temperature']

        # load SAC hyperparameters
        self.program_emb_dim        = cfg['SAC']['hidden_size']
        self.obs_emb_dim            = cfg['SAC']['obs_emb_dim']
        self.replay_buffer_capacity = cfg['SAC']['replay_buffer_capacity']
        self.critic_hidden_dim      = cfg['SAC']['double_q_critic']['hidden_dim']
        self.critic_hidden_depth    = cfg['SAC']['double_q_critic']['hidden_depth']
        self.actor_hidden_dim       = cfg['SAC']['diag_gaussian_actor']['hidden_dim']
        self.actor_hidden_depth     = cfg['SAC']['diag_gaussian_actor']['hidden_depth']
        self.actor_log_std_bounds   = cfg['SAC']['diag_gaussian_actor']['log_std_bounds']
        self.num_train_steps        = cfg['SAC']['num_train_steps']
        self.num_seed_steps         = cfg['SAC']['num_seed_steps']
        self.num_processes          = cfg['SAC']['num_processes']
        self.decoder_deterministic  = cfg['SAC']['decoder_deterministic']
        

        # create karel world for video image generation
        self._world = karel.Karel_world(make_error=False)

        # create log dir
        log_dir = os.path.expanduser(os.path.join(cfg['outdir'], 'SAC', 'openai_SAC'))
        log_dir_best = os.path.expanduser(os.path.join(cfg['outdir'], 'SAC', 'openai_SAC','best'))
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(log_dir_best)
        utils.cleanup_log_dir(eval_log_dir)

        # create env
        cfg_rl = cfg['rl']
        cfg_envs = cfg['rl']['envs']
        custom = True if "karel" or "CartPoleDiscrete" in cfg_envs['executable']['name'] else False
        logger.info('Using environment: {}'.format(cfg_envs['executable']['name']))
        
        assert cfg['mdp_type'] in [
                'ProgramEnv_option_new_vae', 
                'ProgramEnv_option_new_vae_v2', 
                'ProgramEnv_option_new_vae_v2_reconstruct', 
                'ProgramEnv_option_new_vae_v2_key2door_fixed',
                'ProgramEnv_option_new_vae_v2_key2door',
                ]
        self.envs = make_vec_envs(
                cfg_envs['executable']['name'], 
                cfg['seed'], 
                cfg['SAC']['num_processes'], 
                cfg_rl['gamma'], 
                os.path.join(cfg['outdir'], 'SAC', 'openai_SAC'),
                torch.device('cpu'), # replay buffer on cpu
                False, 
                custom_env=custom, 
                custom_env_type='program', 
                custom_kwargs={'config': cfg['args']})

      

        self.cnn_shape   = self.envs.observation_space.shape
        # create actor and critic networks
        self.critic = DoubleQCritic(self.cnn_shape, self.obs_emb_dim, self.program_emb_dim, self.critic_hidden_dim, self.critic_hidden_depth).to(self.device)

        self.critic_target = DoubleQCritic(self.cnn_shape, self.obs_emb_dim, self.program_emb_dim, self.critic_hidden_dim, self.critic_hidden_depth).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(self.cnn_shape, self.obs_emb_dim, self.program_emb_dim, self.actor_hidden_dim, self.actor_hidden_depth, self.actor_log_std_bounds).to(self.device)


        # create algo
        self.agent = SACAgent(self.device, self.cfg, self.critic, self.critic_target, self.actor)

        # create rollout buffer
        self.replay_buffer = ReplayBuffer(
                int(self.replay_buffer_capacity),
                self.num_processes,
                self.envs.observation_space.shape,
                self.program_emb_dim,
                self.device
                )

        # load vae from checkpoint
        self.program_vae = ProgramVAE(self.envs, **cfg)
        self.program_vae.to(device)
        checkpt = self.cfg['net']['saved_params_path']
        assert checkpt is not None
        print('Loading params from {}'.format(checkpt))
        params = torch.load(checkpt, map_location=device)
        self.program_vae.load_state_dict(params[0], strict=True)


        self.episode_rewards           = deque(maxlen=20)
        self.episode_programs          = deque(maxlen=20)
        self.episode_len               = deque(maxlen=20)
        self.episode_primitive_len     = deque(maxlen=20)
        self.episode_s_h               = deque(maxlen=20)
        self.episode_primitive_len     = deque(maxlen=20)
        self.episode_gt_s_h            = deque(maxlen=20)
        self.episode_gt_s_h_len        = deque(maxlen=20)
 

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
        #episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        
        obs = self.envs.reset()

        print("sac_trainer obs shape: ", obs.shape)
        print("obs device: ", obs.device)
        
        num_updates = int(self.num_train_steps) // self.num_processes
        num_seed_updates = int(self.num_seed_steps) // self.num_processes

        for j in range(num_updates):
            #if done:
            #    if self.step > 0:
            #        self.logger.log('train/duration',
            #                        time.time() - start_time, self.step)
            #        start_time = time.time()
            #        self.logger.dump(
            #            self.step, save=(self.step > self.cfg.num_seed_steps))


            #    obs = self.env.reset()
            #    self.agent.reset()
            #    done = False
            #    episode_step = 0


            # sample program for data collection
            if j < num_seed_updates:
                z = torch.rand(self.num_processes, self.program_emb_dim).to(self.device)
            else:
                with utils.eval_mode(self.agent):
                    z = self.agent.act(obs.float().to(self.device), sample=True)
            assert z.min() >= -1.0 and z.max() <= 1.0, "z logit range error"
            
            # run training update
            if j >= num_seed_updates:
                update_info_dict = self.agent.update(self.replay_buffer, j)
            
            with torch.no_grad():
                decoder_output = self.program_vae.vae.decoder(None, z, teacher_enforcing=False, deterministic=self.decoder_deterministic, evaluate=False)
                _, pred_programs, pred_programs_len, pred_programs_log_probs, pred_programs_output_logits, eop_pred_programs, eop_output_logits, pred_program_masks, pred_programs_dist_entropy = decoder_output
               

            next_obs, reward, done, infos = self.envs.step(pred_programs)

            for info in infos:
                if 'episode' in info.keys():
                    # TODO: check on how info['episode']['r'] is added
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_programs.append(info['exec_data']['program_str_history'])
                    self.episode_len.append(info['exec_data']['program_step'])
                    self.episode_primitive_len.append(info['exec_data']['primitive_episode_len'])
                    self.episode_s_h.append(info['exec_data']['s_image_h_list'])
                    if self.cfg['env_task'] == 'program':
                        self.episode_gt_s_h.append(info['exec_data']['gt_s_h'])
                        self.episode_gt_s_h_len.append(info['exec_data']['gt_s_h_len'])


            # allow infinite bootstrap
            n_done = [[0.0] if done_ else [1.0] for done_ in done]
            #n_done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            n_done_no_max = n_done
            
            self.replay_buffer.add(obs.to(torch.bool), z.cpu(), reward, next_obs.to(torch.bool), n_done,
                                   n_done_no_max)

            obs = next_obs

            # save for every interval-th episode or for the last epoch
            if j >= num_seed_updates and (j % self.save_interval == 0 or j == num_updates - 1):
                save_path = os.path.join(self.cfg["outdir"], 'SAC')
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass
    
                torch.save(self.critic, 
                        os.path.join(save_path, self.env_name + "_" + self.cfg['env_task'] + "model_critic.pt"))
                print("Saving model to path: ", os.path.join(save_path, self.env_name + "_" + self.cfg['env_task'] + "model_critic.pt"))

                torch.save(self.critic_target,
                        os.path.join(save_path, self.env_name + "_" + self.cfg['env_task'] + "model_critic_target.pt"))
                print("Saving model to path: ", os.path.join(save_path, self.env_name + "_" + self.cfg['env_task'] + "model_critic_target.pt"))
                
                torch.save(self.actor,
                        os.path.join(save_path, self.env_name + "_" + self.cfg['env_task'] + "model_actor.pt"))
                print("Saving model to path: ", os.path.join(save_path, self.env_name + "_" + self.cfg['env_task'] + "model_actor.pt"))


            if j % int(self.log_video_interval) == 0 and j > 0:
                print("Log Videos...")
                video_len = [len(s_h) for s_h in self.episode_s_h]
                print("video len mean/min/max {}/{}/{}:".format(np.mean(video_len), np.min(video_len), np.max(video_len)))
                # create video log 
                log_video_dir = os.path.expanduser(os.path.join(self.cfg['outdir'], 'SAC', 'video_{}_update{}'.format(self.cfg['env_task'], j)))
                utils.cleanup_log_dir(log_video_dir)
                save_video_dir_path = os.path.join(self.cfg['outdir'], 'SAC', 'video_{}_update{}'.format(self.cfg['env_task'], j))
                for idx, s_h in enumerate(self.episode_s_h):
                    save_video_path = os.path.join(save_video_dir_path, "{}_update{}_sample{}.gif".format(self.cfg['env_task'], j, idx)),
                    self.save_gif(save_video_path[0], s_h)
                    # log program
                for s_idx, (ri, pi) in enumerate(zip(self.episode_rewards, self.episode_programs)):
                    self.writer.add_text(
                            'program/Num_update{}_Sample{}'.format(j, s_idx), 
                            'reward_env: {} program: {} '.format(ri, pi), 
                            total_num_steps
                            )
  
            if j % self.log_interval == 0 and len(self.episode_rewards) > 1:
                total_num_steps = (j + 1) * self.num_processes 
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {}".format(j, total_num_steps, int(total_num_steps / (end - start_time))), end=' ')
                print("Last {} training episodes: mean/min/max reward {:.4f}/{:.4f}/{:.4f}".format(len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards)), end=' ')
                print("Last {} episode len mean/median {:.1f}/{:.1f}".format(len(self.episode_len), np.mean(self.episode_len), np.median(self.episode_len)), end=' ')
                print("Last {} primitive len mean/median {:.1f}/{:.1f}".format(len(self.episode_primitive_len), np.mean(self.episode_primitive_len), np.median(self.episode_primitive_len)), end=' ')
                if self.cfg['env_task'] == 'program':
                    print("Last {} gt demo len mean/median {:.1f}/{:.1f}".format(len(self.episode_gt_s_h_len), np.mean(self.episode_gt_s_h_len), np.median(self.episode_gt_s_h_len)) )
                else:
                    print(' ')


                if j >= num_seed_updates:
                    print("Critic Q1 {:.4f}, Q2 {:.4f}".format(update_info_dict['critic_info']['q1'].mean().item(), update_info_dict['critic_info']['q2'].mean().item()))
                    # Add logs to TB
                    self.writer.add_scalar('train/FPS', int(total_num_steps / (end - start_time)), total_num_steps)
                    self.writer.add_scalar('train/mean_episode_reward', np.mean(self.episode_rewards), total_num_steps)
                    self.writer.add_scalar('train/max_episode_reward', np.max(self.episode_rewards), total_num_steps)
                    self.writer.add_scalar('train/min_episode_reward', np.min(self.episode_rewards), total_num_steps)
                    self.writer.add_scalar('train/mean_episode_len', np.mean(self.episode_len), total_num_steps)
                    self.writer.add_scalar('train/mean_episode_primitive_len', np.mean(self.episode_primitive_len), total_num_steps)
                    self.writer.add_scalar('train/batch_reward', update_info_dict['train/batch_reward'], total_num_steps)
                    self.writer.add_scalar('train_critic/loss', update_info_dict['critic_info']['train_critic/loss'], total_num_steps)
                    self.writer.add_scalar('train_critic/q1', update_info_dict['critic_info']['q1'].mean().item(), total_num_steps)
                    self.writer.add_scalar('train_critic/q2', update_info_dict['critic_info']['q2'].mean().item(), total_num_steps)
                    if self.cfg['env_task'] == 'program':
                        self.writer.add_scalar('train/mean_gt_demo_len', np.mean(self.episode_gt_s_h_len), total_num_steps)
                    
                    
                    if j % self.log_histogram_interval == 0 and j > 0:
                        self.writer.add_histogram('q1', update_info_dict['critic_info']['q1'], total_num_steps)
                        self.writer.add_histogram('q2', update_info_dict['critic_info']['q2'], total_num_steps)
                        if j % int(self.log_video_interval) != 0 and j > 0:
                            # log program
                            for s_idx, (ri, pi) in enumerate(zip(self.episode_rewards, self.episode_programs)):
                                self.writer.add_text(
                                        'program/Num_update{}_Sample{}'.format(j, s_idx), 
                                        'reward_env: {} program: {} '.format(ri, pi), 
                                        total_num_steps
                                        )
 
                    if 'actor_info' in update_info_dict:
                        self.writer.add_scalar('train_actor/loss', update_info_dict['actor_info']['train_actor/loss'], total_num_steps)
                        self.writer.add_scalar('train_actor/target_entropy', update_info_dict['actor_info']['train_actor/target_entropy'], total_num_steps)
                        self.writer.add_scalar('train_actor/entropy', update_info_dict['actor_info']['train_actor/entropy'], total_num_steps)
                        self.writer.add_scalar('train_actor/mu', update_info_dict['actor_info']['mu'].mean().item(), total_num_steps)
                        self.writer.add_scalar('train_actor/std', update_info_dict['actor_info']['std'].mean().item(), total_num_steps)

                        if j % self.log_histogram_interval == 0 and j > 0:
                            self.writer.add_histogram('mu', update_info_dict['actor_info']['mu'], total_num_steps)
                            self.writer.add_histogram('std', update_info_dict['actor_info']['std'], total_num_steps)
                        if self.learnable_temperature:
                            self.writer.add_scalar('train_alpha/loss', update_info_dict['actor_info']['train_alpha/loss'], total_num_steps)
                            self.writer.add_scalar('train_alpha/value', update_info_dict['actor_info']['train_alpha/value'], total_num_steps)
                            

