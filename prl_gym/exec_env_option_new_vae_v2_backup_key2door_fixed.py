"""
This file defines functions for rewarding a synthesized programmatic policy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

sys.path.insert(0, '.')
sys.path.insert(0, 'karel_env')

import time
import numpy as np
import collections
from multiprocessing import Pool
import gym
import prl_gym

from karel_env import karel_option_key2door as karel
from karel_env.dsl import get_DSL_option_v2
from karel_env.dsl.dsl_parse import parse
from dsl.dsl_parse_and_trace import parse_and_trace
from karel_env.generator_option_key2door import KarelStateGenerator


def array_to_str(state):
    return "".join(list(map(str, state.flatten())))


def _branch_execution_ratio(record_dict):
    if len(record_dict) == 0:
        return None

    total_branches = 2 * len(record_dict)
    executed_branches = 0
    for key, value in record_dict.items():
        branch_dict = value[0][1]
        executed_branches += int(branch_dict[True]) + int(branch_dict[False])
    return executed_branches / total_branches



class ExecEnv(object):
    """Custom Environment: given a program, generate reward for current task"""

    def __init__(self, config):
        """Initialize karel state generator, karel world.
        Generate initial state and execution traces for RL task
        """
        self.config = config
        if self.config.env_name == "karel":
            self.dsl = get_DSL_option_v2(dsl_type='prob', seed=config.seed, environment=self.config.env_name)
            self.s_gen = KarelStateGenerator(seed=config.seed)
            self._world = karel.Karel_world(make_error=False, env_task=config.env_task,
                                            task_definition=config.task_definition, reward_diff=config.reward_diff,
                                            final_reward_scale=config.final_reward_scale)
        else:
            raise NotImplementedError('{} not implemented for PRL setup'.format(self.config.env_name))

    def execute_pred_program(self, program_seq, demo=None, demo_len=None):
        raise NotImplementedError

    def reward(self, pred_program_seq):
        """Reward for synthesized programmatic policy (predicted program)"""
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='init_states'):
        if mode == 'init_states':
            return [x[0] for x in self.init_states]
        else:
            return self._world.render(mode)


class ExecEnv1(ExecEnv):
    """Custom Environment: given a program, generate reward for current task"""

    def __init__(self, config, program_seq=None, metadata={}):
        """Initialize karel state generator, karel world.
        Generate initial state and execution traces for RL task
        """
        self.config = config
        super(ExecEnv1, self).__init__(config)
        
        # Note:
        # 1. num_demo_per_program == 1
        # 2. cover_all_branches_in_demos = False
        # 3. env_task == 'program'
        # 4. self.config.experiment == 'intention_space'
        print("num_demo_per_program: ", config.num_demo_per_program) 
        if program_seq is None:
            with open(config.task_file, 'r') as f:
                print(config.task_file)
                config_task_str = f.readlines()[0].strip()
            if config_task_str.split(' ')[0] != 'DEF':
                print(config.task_file)
                config_task_seq = np.loadtxt(config.task_file, dtype=np.int8)
                config_task_str = self.dsl.intseq2str(config_task_seq)
            exe, s_exe = parse(config_task_str, environment='karel')
            if not s_exe:
                assert 0, "specified task doesn't have valid DSL"
            self.gt_program_seq = self.dsl.str2intseq(config_task_str)
        else:
            self.gt_program_seq = program_seq

        # check if task definition is program even if state based reward tasks
        if config.env_task != 'program':
            assert config.task_definition == 'program'
            assert 'tasks/task' not in config.task_file

        init_states = metadata.get('init_states', None)
        self.generate_initial_states = True if init_states is None else False
        if init_states is not None:
            self.init_states = init_states

        gt_exec_data = metadata.get('task_exec_data', None)
        if gt_exec_data is None:
            self.gt_program = self._execute_gt_program(config, self.gt_program_seq)
        else:
            demos_s_h, len_s_h = gt_exec_data
            gt_program = {}
            gt_program['program'] = program_seq
            gt_program['s_h_len'] = len_s_h
            gt_program['s_h'] = demos_s_h
            gt_program['a_h_len'] = None
            gt_program['a_h'] = None
            gt_program['max_demo_length'] = np.max(len_s_h)
            gt_program['dsl_type'] = 'prob'
            gt_program['num_program_tokens'] = len(self.dsl.int2token)
            gt_program['num_demo_per_program'] = config.num_demo_per_program
            gt_program['num_action_tokens'] = len(self.dsl.action_functions)
            self.gt_program = gt_program
            self._world.set_new_state(self.gt_program['s_h'][0])


        if init_states is None:
            self.init_states = np.expand_dims(np.array([demo[0] for demo in self.gt_program['s_h']]), axis=1)
        
        # use fixed input for training 
        self.cnn_input_states = []
        for i in range(config.max_episode_steps + 1):
           s, _, _, _, _ = self.s_gen.generate_program_instruct_state(config.input_height, config.input_width, config.wall_prob, idx=i)
           self.cnn_input_states.append(s)
        print("cnn input state length: ", len(self.cnn_input_states))

        # get max episode step
        self.max_episode_steps    = config.max_episode_steps
        self.fixed_input          = config.fixed_input
        self.max_pred_demo_length = config.max_pred_demo_length
        self.program_step         = 0
        self.progress_ratio       = 0.0
        self.pred_demo            = np.array([[demo[0]]*self.max_pred_demo_length for demo in self.gt_program['s_h']])
        self.pred_demo_length     = np.array([1 for demo in self.gt_program['s_h']])
        self.program_str_history  = []
        self.init_state = self.cnn_input_states[0]
        self.reset_count = 0
        self.gt_sample_demo_period = config.gt_sample_demo_period

    def _execute_gt_program(self, config, program_seq):
        """ Generate valid execution traces for ground-truth program

        Parameters:
            :param config(argparse): hyper-parameters for generating traces
            :param program_seq(np.array): RL task (ground-truth program) in integer sequence format

        Returns: dict
            :return: ground-truth program with execution traces.

        """
        gt_code = self.dsl.intseq2str(program_seq)
        h = config.input_height
        w = config.input_width
        c = len(karel.state_table)
        wall_prob = config.wall_prob

        # parse program to enable execution tracing
        if config.cover_all_branches_in_demos:
            exe, s_exe, record_dict = parse_and_trace(gt_code, environment='karel')
            if len(record_dict) == 0 and ('WHILE' in gt_code or 'IF' in gt_code):
                assert 0, 'only non-conditional programs will have empty dict'
            prev_exec_ratio = exec_ratio = _branch_execution_ratio(record_dict)
            assert prev_exec_ratio == 0.0 or prev_exec_ratio is None
            if not s_exe:
                raise RuntimeError('If we reach here, then we should be able to parse the program')

        s_h_list = []
        a_h_list = []
        num_demo = 0
        num_trial = 0
        num_err_trial = 0
        max_demo_generation_trial = 1000
        while num_demo < config.num_demo_per_program and num_trial < max_demo_generation_trial:
            try:
                if self.generate_initial_states:
                    if self.config.env_task == 'program':
                        s, _, _, _, _ = self.s_gen.generate_single_state(h, w, wall_prob)
                    elif config.env_task == 'cleanHouse' or config.env_task == 'cleanHouse_sparse':
                        s, _, _, _, _ = self.s_gen.generate_single_state_clean_house(h, w, wall_prob)
                    elif config.env_task == 'harvester' or config.env_task == 'harvester_sparse':
                        s, _, _, _, _ = self.s_gen.generate_single_state_harvester(h, w, wall_prob)
                    elif config.env_task == 'fourCorners' or config.env_task == 'fourCorners_sparse':
                        s, _, _, _, _ = self.s_gen.generate_single_state_four_corners(h, w, wall_prob)
                    elif config.env_task == 'randomMaze' or config.env_task == 'randomMaze_sparse':
                        s, _, _, _, _ = self.s_gen.generate_single_state_random_maze(h, w, wall_prob)
                    elif config.env_task == 'stairClimber' or config.env_task == 'stairClimber_sparse':
                        s, _, _, _, _ = self.s_gen.generate_single_state_stair_climber(h, w, wall_prob)
                    elif config.env_task == 'topOff' or config.env_task == 'topOff_sparse':
                        s, _, _, _, _ = self.s_gen.generate_single_state_chain_smoker(h, w, wall_prob, is_top_off=True)
                    else:
                        raise NotImplementedError('{} task not implemented yet'.format(config.env_task))
                else:
                    s = self.init_states[num_demo][0]
                self._world.set_new_state(s)
                if not config.cover_all_branches_in_demos:
                    s_h = self.dsl.run(self._world, gt_code)
                else:
                    self._world.clear_history()
                    self._world, n, s_run = exe(self._world, 0, record_dict, exe)
                    if not s_run:
                        raise RuntimeError("Program execution timeout.")
                    s_h = self._world.s_h
            except RuntimeError:
                num_err_trial += 1
                pass
            else:
                if not config.cover_all_branches_in_demos:
                    if config.max_demo_length >= len(self._world.s_h) >= config.min_demo_length:
                        # we expect to return execution traces in (input, ...., output) format for EGPS
                        # if no actions were executed in environment, repeat initial state and add dummy action for it
                        if len(self._world.a_h) < 1 and config.execution_guided:
                            assert len(self._world.s_h) == 1
                            self._world.a_h.append(self._world.num_actions)
                            self._world.s_h.append(self._world.s_h[0])
                        s_h_list.append(np.stack(self._world.s_h, axis=0))
                        a_h_list.append(np.array(self._world.a_h))
                        num_demo += 1
                else:
                    exec_ratio = _branch_execution_ratio(record_dict)
                    if len(self._world.s_h) <= config.max_demo_length and \
                            (len(self._world.s_h) >= config.min_demo_length or (exec_ratio is not None and (exec_ratio > prev_exec_ratio or (exec_ratio == 1.0 and np.random.uniform() < 0.5 and len(self._world.s_h) >= 2)))) and \
                            (exec_ratio is None or exec_ratio > prev_exec_ratio or exec_ratio >= 1.0):
                        s_h_list.append(np.stack(self._world.s_h, axis=0))
                        a_h_list.append(np.array(self._world.a_h))
                        prev_exec_ratio = exec_ratio
                        num_demo += 1

            num_trial += 1

        if num_demo < config.num_demo_per_program and 'maze' not in config.env_task:
            if config.cover_all_branches_in_demos and exec_ratio is not None and exec_ratio <= 1.0:
                if self.config.env_task == 'program':
                    print("WARNING: could generate {} with only {}% coverage of GT program {}".format(config.num_demo_per_program, exec_ratio*100, gt_code))
                else:
                    assert 0, "couldn't generate {} with {}% coverage demonstrations with 100% coverage for GT {}".format(config.num_demo_per_program, exec_ratio*100, gt_code)

        # np.ndarray for all execution states
        len_s_h = np.array([s_h.shape[0] for s_h in s_h_list], dtype=np.int16)
        demos_s_h = np.zeros([num_demo, np.max(len_s_h), h, w, c], dtype=bool)
        for i, s_h in enumerate(s_h_list):
            demos_s_h[i, :s_h.shape[0]] = s_h

        # np.ndarray for all execution actions
        len_a_h = np.array([a_h.shape[0] for a_h in a_h_list], dtype=np.int16)
        demos_a_h = np.zeros([num_demo, max(np.max(len_a_h), 1)], dtype=np.int8)
        for i, a_h in enumerate(a_h_list):
            demos_a_h[i, :a_h.shape[0]] = a_h

        # save the state
        gt_program = {}
        gt_program['program'] = program_seq
        gt_program['s_h_len'] = len_s_h
        gt_program['a_h_len'] = len_a_h
        gt_program['s_h'] = demos_s_h
        gt_program['a_h'] = demos_a_h
        gt_program['max_demo_length'] = np.max(len_s_h)
        gt_program['dsl_type'] = 'prob'
        gt_program['num_program_tokens'] = len(self.dsl.int2token)
        gt_program['num_demo_per_program'] = config.num_demo_per_program
        gt_program['num_action_tokens'] = len(self.dsl.action_functions)
        return gt_program

    def execute_pred_program(self, program_seq, demo=None, demo_len=None):
        """Generate execution traces for predicted program, given initial state from ground-truth traces

        Parameters:
            :param program_seq: (np.array) RL task (ground-truth program) in integer sequence format
            :param demo: (np.ndarray) ground-truth execution trace
            :param demo_len: (np.ndarray) length of each ground-truth execution

        Returns: dict
            :return: predicted program with execution traces.

        """
        demo = self.gt_program['s_h'] if demo is None else demo
        demo_len = self.gt_program['s_h_len'] if demo_len is None else demo_len
        num_demo = demo_len.shape[0]
        h = demo.shape[2]
        w = demo.shape[3]
        c = len(karel.state_table)
        s_h_list = []
        a_h_list = []
        pred_program = {}
        
        program_str = self.dsl.intseq2str(program_seq)
        self.program_str_history.append(program_str)

        exe, s_exe = parse(program_str, environment='karel')
        self.program_step += 1
        if not s_exe or not len(program_seq) > 4:
            self.program_str_history.append('__prev_invalid__')
            # can't execute the program or it's a dummy program: DEF run m()m
            syntax = False
            demo_correctness = np.array([False] * num_demo)
            num_correct = 0

            # return initial state as initial and final state of demo
            demos_s_h = np.zeros([num_demo, 2, h, w, c], dtype=bool)
            # default no-op will be action = self._world.num_actions+1
            demos_a_h = (self._world.num_actions) * np.ones([num_demo, 1], dtype=np.int8)
            for k in range(num_demo):
                demos_s_h[k, 0], demos_s_h[k, 1] = demo[k][0], demo[k][0]

            # save the state
            pred_program['s_h_len'] = 2 * np.ones(num_demo, dtype=np.int16)
            pred_program['a_h_len'] = 1 * np.ones(num_demo, dtype=np.int16)
            pred_program['s_h'] = demos_s_h
            pred_program['a_h'] = demos_a_h
            pred_program['max_demo_length'] = np.max(pred_program['s_h_len'])
        else:
            syntax = True
            demo_correctness = np.array([False] * num_demo)
            for k in range(num_demo):
                init_state_idx = self.pred_demo_length[k] - 1
                init_state = self.pred_demo[k][init_state_idx]
                if self.program_step == 1:
                    assert np.all(init_state == demo[k][0])
                    
                self._world.clear_history()
                self._world.set_new_state(init_state)
                exe, s_exe = parse(program_str)
                if not s_exe:
                    raise RuntimeError('This should be correct')

                self._world, n, s_run = exe(self._world, 0)

                # we expect to return execution traces in (input, ...., output) format for EGPS
                # if no actions were executed in environment, repeat initial state and add dummy action for it
                if len(self._world.a_h) < 1 and self.config.execution_guided:
                    assert len(self._world.s_h) == 1
                    self._world.a_h.append(self._world.num_actions)
                    self._world.s_h.append(self._world.s_h[0])
                
                exe_result_len = len(self._world.s_h)
                exe_result = np.stack(self._world.s_h)
                
                
                if self.pred_demo_length[k] + (exe_result_len - 1) < self.max_pred_demo_length:
                    self.pred_demo[k][self.pred_demo_length[k] : self.pred_demo_length[k] + (exe_result_len - 1)] = exe_result[1:]
                    self.pred_demo_length[k] += (exe_result_len - 1) # init state is repeated
                #else:
                #    print("Warning: pred_demo_length > max_pred_demo_length") 

                #demo_correctness[k] = (demo_len[k] == exe_result_len and np.all(demo[k][:demo_len[k]] == exe_result))
                demo_correctness[k] = (demo_len[k] == self.pred_demo_length[k] and np.all(demo[k][:demo_len[k]] == self.pred_demo[k][:self.pred_demo_length[k]]))

                s_h_list.append(np.stack(self._world.s_h, axis=0))
                a_h_list.append(np.array(self._world.a_h))
            num_correct = demo_correctness.astype(np.int32).sum()

            # np.ndarray for all execution states
            len_s_h = np.array([s_h.shape[0] for s_h in s_h_list], dtype=np.int16)
            demos_s_h = np.zeros([num_demo, np.max(len_s_h), h, w, c], dtype=bool)
            for i, s_h in enumerate(s_h_list):
                demos_s_h[i, :s_h.shape[0]] = s_h

            # np.ndarray for all execution actions
            len_a_h = np.array([a_h.shape[0] for a_h in a_h_list], dtype=np.int16)
            demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=np.int8)
            for i, a_h in enumerate(a_h_list):
                demos_a_h[i, :a_h.shape[0]] = a_h

            # save the state
            pred_program['s_h_len'] = len_s_h
            pred_program['a_h_len'] = len_a_h
            pred_program['s_h'] = demos_s_h
            pred_program['a_h'] = demos_a_h
            pred_program['max_demo_length'] = np.max(len_s_h)

        # save the state
        final_state_idx = self.pred_demo_length[0] - 1
        final_state = self.pred_demo[0][final_state_idx]
        pred_program['program'] = program_seq
        pred_program['current_state'] = self.cnn_input_states[self.program_step] 
        pred_program['done'] = self.program_step == self.max_episode_steps
        pred_program['program_str_history'] = self.program_str_history
        pred_program['s_image_h_list'] = self.pred_demo[0][:final_state_idx+1]
        pred_program['program_step'] = self.program_step
        pred_program['primitive_episode_len'] = final_state_idx
        pred_program['program_syntax'] = 'correct' if syntax else 'wrong'
        pred_program['num_execution'] = num_demo
        pred_program['program_prediction'] = program_str
        pred_program['program_is_correct_execution'] = demo_correctness
        pred_program['program_num_execution_correct'] = num_correct

        return pred_program

    def _get_first_exact_match_reward(self, pred_program):
        def _compare_demos(demo1, demo1_len, demo2, demo2_len):
            if demo2_len == 0: return 0
            if demo1_len == 1 and demo2_len == 1: return 1
            fr_reward = 0
            for j in range(min(demo1_len, demo2_len)):
                if np.array_equal(demo1[j], demo2[j]):
                    fr_reward += 1
                else:
                    return (fr_reward-1) / (max(demo1_len, demo2_len)-1)
            return (fr_reward-1) / (max(demo1_len, demo2_len)-1)

        assert self.gt_program['s_h_len'].shape[0] == pred_program['s_h_len'].shape[0]
        num_demos, max_demo_len, h, w, c = self.gt_program['s_h'].shape
        r = 0
        for i in range(num_demos):
            r += _compare_demos(self.gt_program['s_h'][i], self.gt_program['s_h_len'][i],
                                pred_program['s_h'][i], pred_program['s_h_len'][i])
        return r

    def reward(self, pred_program_seq):
        """Reward for synthesized programmatic policy (predicted program)"""
        pred_program = self.execute_pred_program(pred_program_seq, self.gt_program['s_h'], self.gt_program['s_h_len'])

        if self.config.reward_type == 'sparse':
            r = pred_program['program_num_execution_correct'] / pred_program['num_execution']
        elif self.config.reward_type == 'extra_sparse':
            r = pred_program['program_num_execution_correct'] // pred_program['num_execution']
        elif self.config.reward_type == 'dense_subsequence_match':
            option_pred_program = {'s_h': self.pred_demo, 's_h_len': self.pred_demo_length}
            current_progress_ratio = self._get_first_exact_match_reward(option_pred_program) / pred_program['num_execution']
            r = current_progress_ratio - self.progress_ratio
            self.progress_ratio = current_progress_ratio
        else:
            raise NotImplementedError

        if self.config.reward_validity and pred_program['program_syntax'] == 'correct': # False
            r += 0.1/self.config.max_program_len

        if self.config.experiment == 'intention_space': # True
            r = -0.001 if pred_program['program_syntax'] != 'correct' else (r + 0.001)
            # r = 0.01 if pred_program['program_syntax'] != 'correct' else (r + 0.1)
            # r = 0.01 if pred_program['program_syntax'] != 'correct' else math.exp(r + 0.1)

        # return ground truth program action history to calculate condition policy rewards
        pred_program['gt_s_h'] = self.gt_program['s_h']
        pred_program['gt_s_h_len'] = self.gt_program['s_h_len']
        pred_program['gt_a_h'] = self.gt_program['a_h']
        pred_program['gt_a_h_len'] = self.gt_program['a_h_len']
        pred_program['reward_h'] = np.array([r])
        
        pred_program['pred_demo'] = self.pred_demo
        pred_program['pred_demo_length'] = self.pred_demo_length

        return r, pred_program

    def reset(self):
        self.reset_count = (self.reset_count + 1) % self.gt_sample_demo_period
        if self.reset_count == 0:
            self.gt_program = self._execute_gt_program(self.config, self.gt_program_seq)
        if self.generate_initial_states:
            self.init_states = np.expand_dims(np.array([demo[0] for demo in self.gt_program['s_h']]), axis=1)
        
        self.program_step = 0
        self.progress_ratio = 0.0
        self.program_str_history = []
        for i, demo in enumerate(self.gt_program['s_h']):
            self.pred_demo[i][0]      = demo[0]
            self.pred_demo_length[i]  = 1
        self.init_state = self.cnn_input_states[0]  
        return self.init_state


class ExecEnv_option(ExecEnv):
    """Custom Environment: given a program, generate reward for current task"""

    def __init__(self, config, metadata={}):
        """Initialize karel state generator, karel world.
        Generate initial state and execution traces for RL task
        """
        self.config = config
        super(ExecEnv_option, self).__init__(config)
        print("ExecEnv_option on init, env_task: ", config.env_task)
        if config.env_task == 'cleanHouse' or config.env_task == 'cleanHouse_sparse':
            self.init_func = self.s_gen.generate_single_state_clean_house
        elif config.env_task == 'harvester' or config.env_task == 'harvester_sparse':
            self.init_func = self.s_gen.generate_single_state_harvester
        elif config.env_task == 'fourCorners' or config.env_task == 'fourCorners_sparse':
            self.init_func = self.s_gen.generate_single_state_four_corners
        elif config.env_task == 'randomMaze' or config.env_task == 'randomMaze_sparse':
            self.init_func = self.s_gen.generate_single_state_random_maze
        elif config.env_task == 'stairClimber' or config.env_task == 'stairClimber_sparse':
            self.init_func = self.s_gen.generate_single_state_stair_climber
        elif config.env_task == 'topOff' or config.env_task == 'topOff_sparse':
            self.init_func = self.s_gen.generate_single_state_chain_smoker
        elif config.env_task == 'randomMaze_key2door' or config.env_task == 'randomMaze_key2door_sparse':
            self.init_func = self.s_gen.generate_single_state_random_maze_key2door
        elif config.env_task == 'randomMaze_key2doorSpace' or config.env_task == 'randomMaze_key2doorSpace_sparse':
            self.init_func = self.s_gen.generate_single_state_random_maze_key2doorSpace
        elif config.env_task == 'oneStroke' or config.env_task == 'oneStroke_sparse':
            self.init_func = self.s_gen.generate_single_state_oneStroke
        elif config.env_task == 'doorkey' or config.env_task == 'doorkey_sparse':
            self.init_func = self.s_gen.generate_single_state_doorkey
        elif config.env_task == 'seeder' or config.env_task == 'seeder_sparse':
            self.init_func = self.s_gen.generate_single_state_seeder
        elif config.env_task == 'snake' or config.env_task == 'snake_sparse':
            self.init_func = self.s_gen.generate_single_state_snake
        else:
            raise NotImplementedError('task not implemented yet')

        self.init_state = None
        if 'topOff' not in config.env_task:
            self.init_state, _, _, _, self.metadata = self.init_func(config.input_height, config.input_width, config.wall_prob)
        else:
            self.init_state, _, _, _, self.metadata = self.init_func(config.input_height, config.input_width, config.wall_prob, is_top_off=True)

        # use fixed input for training
        self.cnn_input_states = []
        for i in range(config.max_episode_steps + 1):
            s, _, _, _, _ = self.s_gen.generate_program_instruct_state(config.input_height, config.input_width, config.wall_prob, idx=i)
            self.cnn_input_states.append(s)
        print("cnn input state length: ", len(self.cnn_input_states))

        assert self.init_state is not None
        self.current_state = self.cnn_input_states[0]
        print("use fixed image input")
        
        self._world.set_new_state(self.init_state, self.metadata)
        self.h, self.w, self.c = self._world.s_h[0].shape
        self.program_str_history = []
        self.s_image_h_list = []
        self.a_image_h_list = []
        self.primitive_s_h_len = 0
        self.primitive_a_h_len = 0
        self.primitive_r_h_len = 0
        assert self._world.elapse_step == self.primitive_s_h_len
        self.program_step = 0

    def execute_pred_program(self, program_seq):
        s_h_list = []
        a_h_list = []
        r_h_list = []
        pred_program = {}
        self.program_step += 1

        # No clear_history, just keep playing
        # clean up s_h, a_h, r_h of previous program
        # self._world.clear_history()

        program_str = self.dsl.intseq2str(program_seq)
        self.program_str_history.append(program_str)

        exe, s_exe = parse(program_str, environment=self.config.env_name)
        if not s_exe or not len(program_seq) > 4:
            #print("Invalid programs: ", program_str)
            self.program_str_history.append('__prev_invalid__')
            # TODO: log invalid pred_program ratio 
            # can't execute the program or it's a dummy program: DEF run m()m
            syntax = False
            
            # return initial state as initial and final state of demo
            demos_s_h = np.zeros([2, self.h, self.w, self.c], dtype=bool)
            # default no-op will be action = self._world.num_actions+1
            demos_a_h = (self._world.num_actions) * np.ones([1], dtype=np.int16)
            demos_s_h[0], demos_s_h[1] = self.init_state, self.init_state

            # save the state
            pred_program['s_h_len'] = [2]
            pred_program['a_h_len'] = [1]
            pred_program['s_h'] = demos_s_h
            pred_program['a_h'] = demos_a_h
            pred_program['reward_h'] = np.zeros([1], dtype=np.float32) - 0.001 # penalty for invalid programs
            pred_program['max_demo_length'] = [2]
            len_r_h = np.array([1], dtype=np.int16)
        else:
           
            syntax = True
            exe, s_exe = parse(program_str, environment=self.config.env_name)
            if not s_exe:
                raise RuntimeError('This should be correct')

            self._world, n, s_run = exe(self._world, 0)
            # we expect to return execution traces in (input, ...., output) format for EGPS
            # if no actions were executed in environment, repeat last state and add dummy action for it
            if len(self._world.a_h) < 1 and self.config.execution_guided:
                assert len(self._world.s_h) == 1
                self._world.s_h.append(self._world.s_h[-1])
                self._world.a_h.append(self._world.num_actions)
                self._world.r_h.append(0.0)

            
            assert self._world.elapse_step >= self.primitive_s_h_len
            s_h_list = np.stack(self._world.s_h, axis=0)[self.primitive_s_h_len : self._world.elapse_step+1]
            a_h_list = np.array(self._world.a_h)[self.primitive_s_h_len : self._world.elapse_step]
            r_h_list = np.array(self._world.r_h)[self.primitive_s_h_len : self._world.elapse_step]
            
            self.primitive_s_h_len = self._world.elapse_step
            self.primitive_a_h_len = self._world.elapse_step
            self.primitive_r_h_len = self._world.elapse_step


            # np.ndarray for all execution states
            len_s_h = [s_h_list.shape[0]]

            # np.ndarray for all execution actions
            len_a_h = [a_h_list.shape[0]]

            # np.ndarray for all execution rewards (reward and action lengths should be same)
            len_r_h = [r_h_list.shape[0]]

            # save the state
            pred_program['s_h_len'] = len_s_h
            pred_program['a_h_len'] = len_a_h
            pred_program['s_h'] = s_h_list
            pred_program['a_h'] = a_h_list
            pred_program['max_demo_length'] = len_s_h
            pred_program['reward_h'] = r_h_list

        # create state image
        if len(self.s_image_h_list) == 0:
            for s in pred_program['s_h']:
                self.s_image_h_list.append(s)
        else:
            for s in pred_program['s_h'][1:]:
                self.s_image_h_list.append(s)
        for a in pred_program['a_h']:
            self.a_image_h_list.append(a)

        # save the state
        #self.current_state = self._world.s.copy()
        pred_program['program'] = program_seq
        pred_program['current_state'] = self.cnn_input_states[self.program_step]
        pred_program['done'] = self._world.done
        pred_program['program_str_history'] = self.program_str_history
        pred_program['s_image_h_list'] = self.s_image_h_list
        pred_program['a_image_h_list'] = self.a_image_h_list
        pred_program['program_step'] = self.program_step
        pred_program['primitive_episode_len'] = self.primitive_a_h_len
        pred_program['program_syntax'] = 'correct' if syntax else 'wrong'
        if self.config.env_task == 'stairClimber':
            pred_program['mean_reward'] = np.sum(pred_program['reward_h']) / max(len_r_h[0], 1)
            if len_r_h[0] == 0:
                pass
                # print("len_r_h[0] == 0, program: ", program_str)
        else:
            pred_program['mean_reward'] = np.sum(pred_program['reward_h'])

        return pred_program

    def reward(self, pred_program_seq):
        """Reward for synthesized programmatic policy (predicted program)"""
        pred_program = self.execute_pred_program(pred_program_seq)
        reward = pred_program['mean_reward']
        return reward, pred_program

    def reset(self):
        self.init_state, _, _, _, self.metadata = self.init_func(self.config.input_height, self.config.input_width, self.config.wall_prob)

        assert self.init_state is not None
        self.current_state = self.cnn_input_states[0]
        self._world.clear_history()
        self._world.set_new_state(self.init_state, self.metadata)
        self.h, self.w, self.c = self._world.s_h[0].shape
        self.program_str_history = []
        self.s_image_h_list = []
        self.a_image_h_list = []
        self.program_step = 0
        #self._world.print_state()
        self.primitive_s_h_len = 0
        self.primitive_a_h_len = 0
        self.primitive_r_h_len = 0
        assert self._world.elapse_step == self.primitive_s_h_len
 
        return self.current_state.copy() #self.init_state.copy()

