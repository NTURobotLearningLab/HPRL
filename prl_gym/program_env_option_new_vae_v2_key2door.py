import time
import numpy as np
import gym
from gym import spaces
from exec_env_option_new_vae_v2_backup_key2door import ExecEnv1, ExecEnv_option
from karel_env.dsl.dsl_parse import parse


class ProgramEnv_option_new_vae_v2(gym.Env):
    """MDP3
        state: karel state (image)
        action: sub-program (option)
        Transition: sub-program -> sub-program
        reward: environment reward for executing the program
        Done: whether reward threshold / max_episode_steps is reached
        info: record action (sub-program history) ...
    """

    def __init__(self, config, task=None, metadata={}):
        super(ProgramEnv_option_new_vae_v2, self).__init__()
        self.metadata = {'render.modes': ['rgb_array', 'program', 'init_states']}
        self.config = config
        self.max_program_len = config.max_program_len
        self._elapsed_steps = 0
        self._episode_reward = 0.0
        self._max_episode_steps = config.max_episode_steps
        self.partial_program = []       
        if self.config.task_definition == 'program':
            self.task_env = ExecEnv1(config, task, metadata)
            self.gt_reward, _ = self.task_env.reward(self.task_env.gt_program_seq)
            print("gt_reward: ", self.gt_reward)
            self.task_env.reset()
        elif self.config.task_definition == 'custom_reward':
            self.gt_reward = 10000.0
            self.task_env = ExecEnv_option(config, metadata)
        else:
            raise NotImplementedError

        print("ProgramEnv_option_new_vae_v2 _max_episode_steps: ", self._max_episode_steps)
        
        # Add one token for invalid token (all tokens after end token should be invalid)
        if config.use_simplified_dsl: # False
            self.num_program_tokens = len(config.prl_tokens)+1
            self.T2I = {tkn: i for i, tkn in enumerate(config.prl_tokens)}
        else:
            self.num_program_tokens = len(self.task_env.dsl.int2token)+1
            self.T2I = {tkn: i for i, tkn in enumerate(config.dsl_tokens)}
        print("ProgramEnv_option_new_vae_v2 num_program_tokens: ", self.num_program_tokens)

        # define action space
        self.alpha = 1
        assert config.action_type == "program"
        self.action_space = spaces.Box(
                low=0, 
                high=self.num_program_tokens,
                shape=(self.alpha*self.max_program_len,), 
                dtype=np.int8
                )

        # define observation space
        if config.obv_type == "program":
            self.initial_obv = self.task_env.init_state.copy()
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.initial_obv.shape), dtype=np.uint8)
        elif config.obv_type == "encoded": 
            #TODO:  may be used for vector_feature
            #self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=[config.num_lstm_cell_units], dtype=np.float32)
            #self.initial_obv = np.zeros(config.num_lstm_cell_units)
            raise NotImplementedError('observation not recognized')
        else:
            raise NotImplementedError('observation not recognized')

        self.state = self.initial_obv

    def _prl_to_dsl(self, program_seq):
        def func(x):
            return self.config.dsl_tokens.index(self.config.prl2dsl_mapping[self.config.prl_tokens[x]])
        return np.array(list(map(func, program_seq)), program_seq.dtype)

    def _set_bad_transition(self, reward, done, info):
        # TODO: need to shift this code under rl.envs.TimeLimitMask
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            info['bad_transition'] = done
            done = True
        return reward, done, info

    def _modify(self, action):
        # Ignore everything after end-of-program token
        null_token_idx = np.argwhere(action == (self.num_program_tokens-1))
        if null_token_idx.shape[0] > 0:
            action = action[:null_token_idx[0].squeeze()]
        # remap prl tokens to dsl tokens if we are using simplified DSL
        action = self._prl_to_dsl(action) if self.config.use_simplified_dsl else action
        return action

    def step(self, action):
        """Currently state is previous program, action is new program
        Alert: action can be in simplified DSL format, make sure to use transformed action
               (here we transform it in _modify())
        """
        assert self.alpha == 1

        self._elapsed_steps += 1
        dsl_action = self._modify(action)
        # FIXME: temporary fix for ignoring DEF, run, )m kind of tokens
        # TODO: Check on this later
        if self.config.experiment == 'intention_space': # True
            dsl_action = np.concatenate((np.array([0]), dsl_action))
        else:
            if self.config.grammar is not None:
                dsl_action = np.concatenate((np.array([0, 1, 2]), dsl_action))
            else:
                dsl_action = np.concatenate((np.array([0, 1, 2]), dsl_action, np.array([3])))

        program_seq = dsl_action
        reward, exec_data = self.task_env.reward(program_seq)
        done = exec_data['done']
        self.state = exec_data['current_state']
        info = {'cur_state': action, 'modified_action': dsl_action, 'exec_data': exec_data}

        reward, done, info = self._set_bad_transition(reward, done, info)
        self._episode_reward += reward
        # call reset if done
        if done:
            info['episode'] = {'r': self._episode_reward, 'p': exec_data['program_str_history']}
            self.state = self.reset()

        return self.state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self._elapsed_steps = 0
        self._episode_reward = 0.0
        self.partial_program = []
        self.initial_obv = self.task_env.reset()
        self.state = self.initial_obv
        return self.initial_obv

    def render(self, mode='init_states'):
        """render current program for a random initial state"""
        #if mode == 'program':
        #    pred_program = self.task_env.execute_pred_program(self.state)
        #    return pred_program
        #elif mode == 'init_states':
        #    return self.task_env.render(mode='init_states')
        #else:
        #    raise NotImplementedError('Yet to generate video of predicted program execution')
        raise NotImplementedError('Yet to generate video of predicted program execution')
        
