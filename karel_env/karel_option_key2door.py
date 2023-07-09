import os
import numpy as np
import scipy
from scipy import spatial
from collections import deque

MAX_NUM_MARKER_topoff = 2
MAX_NUM_MARKER_general = 1

state_table = {
    0: 'Karel facing North',
    1: 'Karel facing East',
    2: 'Karel facing South',
    3: 'Karel facing West',
    4: 'Wall',
    5: '0 marker',
    6: '1 marker',
    7: '2 markers',
}

action_table = {
    0: 'Move',
    1: 'Turn left',
    2: 'Turn right',
    3: 'Pick up a marker',
    4: 'Put a marker'
}


class Karel_world(object):

    def __init__(self, s=None, make_error=True, env_task="program", task_definition='program' ,reward_diff=False, final_reward_scale=True):
        if s is not None:
            self.set_new_state(s)
        self.make_error = make_error
        self.env_task = env_task
        self.task_definition = task_definition
        self.rescale_reward = True
        self.final_reward_scale = final_reward_scale
        self.reward_diff = reward_diff
        self.num_actions = len(action_table)
        self.elapse_step = 0
        self.progress_ratio = 0.0

    def set_new_state(self, s, metadata=None):
        self.elapse_step = 0
        self.perception_count = 0
        self.progress_ratio = 0.0
        self.s = s.astype(np.bool)
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.h = self.s.shape[0]
        self.w = self.s.shape[1]
        p_v = self.get_perception_vector()
        self.p_v_h = [p_v.copy()]
        self.pos_h = [tuple(self.get_location()[:2])]
        self.pos_h_set = set(self.pos_h)
        self.snake_body = deque([(1, 1), (1, 2)])
        self.snake_len  = 2

        if self.task_definition != "program":
            self.r_h = []
            self.d_h = []
            self.progress_h = []
            self.program_reward = 0.0
            self.prev_pos_reward = 0.0
            self.init_pos_reward = 0.0
            self.done = False
            # self.stage = 0 # For key2door
            self.metadata = metadata
            self.total_markers = np.sum(s[:,:,6:])
            if self.env_task == 'snake':
                self.snake_marker_pointer = self.metadata['marker_pointer']
                self.snake_marker_list = self.metadata['marker_list']
            #print(self.snake_marker_list)
            #print(self.snake_marker_pointer)
    ###################################
    ###    Collect Demonstrations   ###
    ###################################

    def clear_history(self):
        self.perception_count = 0
        self.elapse_step = 0
        self.progress_ratio = 0.0
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.p_v_h = []
        self.pos_h = [tuple(self.get_location()[:2])]
        self.pos_h_set = set(self.pos_h)
        self.snake_body = deque([(1, 1), (1, 2)])
        self.snake_len  = 2

        if self.task_definition != "program":
            self.r_h = []
            self.progress_h = []
            self.d_h = []
            self.program_reward = 0.0
            self.prev_pos_reward = 0.0
            self.init_pos_reward = 0.0
            self.done = False
            self.total_markers = np.sum(self.s_h[-1][:,:,6:])

    def add_to_history(self, a_idx, agent_pos, made_error=False):
        self.s_h.append(self.s.copy())
        self.a_h.append(a_idx)
        p_v = self.get_perception_vector()
        self.p_v_h.append(p_v.copy())

        self.elapse_step += 1

        if self.task_definition != "program":
            reward, done = self._get_state_reward(agent_pos, made_error)
            # log agent position
            pos_tuple = tuple(agent_pos[:2])
            self.pos_h_set.add(pos_tuple)
            self.pos_h.append(pos_tuple)           
            
            self.done = self.done or done
            self.r_h.append(reward)
            self.progress_h.append(self.progress_ratio)
            self.d_h.append(done)
            self.program_reward += reward

        self.total_markers = np.sum(self.s[:,:,6:]) 
        #if self.task_definition != 'program' and not made_error:
        #    if a_idx == 3: self.total_markers -= 1
        #    if a_idx == 4: self.total_markers += 1

    def _get_cleanHouse_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        pick_marker = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        for mpos in self.metadata['marker_positions']:
            if state[mpos[0], mpos[1], 5] and not state[mpos[0], mpos[1], 6]:
                pick_marker += 1

        current_progress_ratio = pick_marker / float(len(self.metadata['marker_positions']))
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = pick_marker == len(self.metadata['marker_positions'])

        reward = reward if self.env_task == 'cleanHouse' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_harvester_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # calculate total_1 marker in the state
        max_markers = (w-2)*(h-2)

        assert max_markers >= self.total_markers, "max_marksers: {}, self.total_markers: {}".format(max_markers, self.total_markers)
        
        current_progress_ratio = (max_markers - self.total_markers) / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = self.total_markers == 0

        reward = reward if self.env_task == 'harvester' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_randomMaze_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        distance_to_goal = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 1: assert 0, '{} markers found!'.format(len(x))
        marker_pos = np.asarray([x[0], y[0]])
        distance_to_goal = -1 * spatial.distance.cityblock(agent_pos[:2], marker_pos)

        done = distance_to_goal == 0
        reward = float(done)
        self.done = self.done or done
        return reward, done

    def _get_fourCorners_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        correct_markers = 0
        reward = 0.0
        
        assert not done and not self.done

        # calculate correct markers
        if state[1, 1, 6]:
            correct_markers += 1
        if state[h-2, 1, 6]:
            correct_markers += 1
        if state[h-2, w-2, 6]:
            correct_markers += 1
        if state[1, w-2, 6]:
            correct_markers += 1
        
        self.total_markers = np.sum(self.s[:,:,6:]) 
        assert self.total_markers >= correct_markers, "total_markers: {}, correct_markers: {}".format(self.total_markers, correct_markers)
        #give zero reward if agent places marker anywhere else
        incorrect_markers = self.total_markers - correct_markers
       
        current_progress_ratio = correct_markers / 4.0
        #if current_progress_ratio > self.progress_ratio:
        #    reward = current_progress_ratio - self.progress_ratio
        #    self.progress_ratio = current_progress_ratio
            
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio


        if incorrect_markers > 0 and reward > 0.0:
            reward = 0.0

        done = correct_markers == 4 or incorrect_markers > 0 
        
        if self.env_task == 'fourCorners_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_stairClimber_task_reward(self, agent_pos):
        # check if already done
        assert self.reward_diff == True

        if self.done:
            return 0.0, self.done

        done = False
        state = self.s_h[-1]

        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 1: assert 0, '{} markers found!'.format(len(x))
        marker_pos = np.asarray([x[0], y[0]])
        reward = -1 * spatial.distance.cityblock(agent_pos[:2], marker_pos)
        
        # initial agent position
        x, y, z = np.where(self.s_h[0][:, :, :4] > 0)
        init_pos = np.asarray([x[0], y[0], z[0]])
        longest_distance = spatial.distance.cityblock(init_pos[:2], marker_pos)
        assert longest_distance >= 1.0

        # NOTE: need to do this to avoid high negative reward for first action
        if len(self.s_h) == 2:
            x, y, z = np.where(self.s_h[0][:, :, :4] > 0)
            init_pos = np.asarray([x[0], y[0], z[0]])
            self.prev_pos_reward = -1 * spatial.distance.cityblock(init_pos[:2], marker_pos)

        if not self.reward_diff:
            # since reward is based on manhattan distance, rescale it to range between 0 to 1
            if self.rescale_reward:
                from_min, from_max, to_min, to_max = -(sum(self.s.shape[:2])), 0, -1, 0
                reward = ((reward - from_min) * (to_max - to_min) / (from_max - from_min)) + to_min
            if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions']:
                reward = -0.2 # -1.0
            done = reward == 0
        else:
            abs_reward = reward
            reward = self.prev_pos_reward-1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
            reward = (reward - self.prev_pos_reward) / longest_distance
            assert reward < 1.0, "agent pos: {}, marker_pos: {}, reward: {}, prev_pos_reward: {}".format(agent_pos[:2], marker_pos, reward, self.prev_pos_reward)
            #reward = -1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
            self.prev_pos_reward = abs_reward
            done = abs_reward == 0

        # # calculate previous distance to the goal
        # # TODO: check why this work
        # x, y, z = np.where(self.s_h[-2][:, :, :4] > 0)
        # prev_pos = np.asarray([x[0], y[0], z[0]])
        # self.prev_pos_reward = -1 * spatial.distance.cityblock(prev_pos[:2], marker_pos)


        # current_progress_ratio = (distance_to_goal - self.prev_pos_reward) / (self.init_pos_reward * -1)
        # reward = current_progress_ratio - self.progress_ratio
        # self.progress_ratio = current_progress_ratio
        # reward = -1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
        # self.prev_pos_reward = distance_to_goal
        # done = distance_to_goal == 0

        reward = reward if self.env_task == 'stairClimber' else float(done)
        if self.env_task == 'stairClimber_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_topOff_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done       

        assert self.reward_diff
        done = False
        score = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        for c in range(1, agent_pos[1]+1):
            if (h-2, c) in self.metadata['not_expected_marker_positions']:
                if state[h-2, c, 7]:
                    score += 1
                else:
                    break
            else:
                assert (h-2, c) in self.metadata['expected_marker_positions']
                if state[h-2, c, 5]:
                    score += 1
                else:
                    break

        if (self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0]) and score == w-2:
            score += 1

        current_progress_ratio = score / (w-1)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = sum([state[pos[0], pos[1], 7] for pos in self.metadata['not_expected_marker_positions']]) == len(
            self.metadata['not_expected_marker_positions']) and (self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0]) and current_progress_ratio==1.0

        reward = reward if self.env_task == 'topOff' else float(done)
        if self.env_task == 'topOff_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done


    def _get_randomMaze_key2door_task_reward(self, agent_pos): ## key2door
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        '''
        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 2: assert 0, '{} markers found!'.format(len(x))

        mxs, mys = np.where(state[:, :, 6] > 0)
        if len(mxs):
            for mx, my in zip(mxs, mys):
                if (mx, my) != (x[0], y[0]) and (mx, my) != (x[1], y[1]):
                    self.done = 1
                    if self.stage == 0:
                        return -0.1, self.done
                    else:
                        return -0.1, self.done

        prev_stage = self.stage
        if self.stage == 0:
            if state[x[0], y[0], 7] or state[x[1], y[1], 7]:
                self.done = 1
                return -0.1, self.done
            elif len(mxs) < 2:
                self._door = (mxs[0], mys[0])
                self._key = (x[0], y[0]) if (self._door == (x[1], y[1])) else (x[1], y[1])
                self.stage = 1
        elif self.stage == 1:
            if not state[self._key[0], self._key[1], 5]:
                self.done = 1
                return -0.1, self.done
            if state[self._door[0], self._door[1], 5]:
                self.done = 1
                return -0.1, self.done
            if state[self._door[0], self._door[1], 7]:
                self.stage = 2
                done = True
        '''
        total_markers = np.sum(state[:,:,6:])
        error_markers = total_markers - 2
        score = 0
        if state[6, 3, 5]: # [1, 3, 7]
            score += 0.5
        if state[6, 3, 5] and state[1, 6, 7]:
            score += 0.5
        
        #for y in range(1, 6):
        #    if state[1, y, 6] or state[1, y, 7]:
        #        score -= 0.1
        #for x in range(2, 6):
        #    if state[x, 3, 6] or state[x, 3, 7]:
        #        score -= 0.1
        if error_markers > 0:
            score -= error_markers * 0.0001

        #if state[6, 3, 7]:
        #    score -= 0.1
        #if state[1, 6, 5]:
        #    score -= 0.1

        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (current_progress_ratio==1.0)

        reward = reward if self.env_task == 'randomMaze_key2door' else float(done)
        if self.env_task == 'randomMaze_key2door_sparse':
            reward = reward if done and not self.done else 0
                
        self.done = self.done or done
        return reward, done

   
    def _get_randomMaze_key2doorSpace_task_reward(self, agent_pos): 
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        total_markers = np.sum(state[:,:,6:])
        error_markers = total_markers - 2
        score = 0
        if state[6, 3, 5]: # [1, 3, 7]
            score += 0.5
        if state[6, 3, 5] and state[1, 6, 7]:
            score += 0.5
        
        #for y in range(1, 6):
        #    if state[1, y, 6] or state[1, y, 7]:
        #        score -= 0.1
        #for x in range(2, 6):
        #    if state[x, 3, 6] or state[x, 3, 7]:
        #        score -= 0.1
        if error_markers > 0:
            score -= error_markers * 0.0001 # penalty

        #if state[6, 3, 7]:
        #    score -= 0.1
        #if state[1, 6, 5]:
        #    score -= 0.1

        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (current_progress_ratio==1.0)

        reward = reward if self.env_task == 'randomMaze_key2doorSpace' else float(done)
        if self.env_task == 'randomMaze_key2doorSpace_sparse':
            reward = reward if done and not self.done else 0
                
        self.done = self.done or done
        return reward, done


    def _get_oneStroke_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done
 
        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        pos_tuple = tuple(agent_pos[:2])
        
        is_overlap  = pos_tuple in self.pos_h_set and pos_tuple != self.pos_h[-1]
        is_hit_wall = self.a_h[-1] == 0 and pos_tuple == self.pos_h[-1]
        traverse_length = len(self.pos_h_set)

        # calculate total_1 marker in the state
        max_markers = (w-2)*(h-2)

        assert max_markers >= self.total_markers, "max_marksers: {}, self.total_markers: {}".format(max_markers, self.total_markers)
        assert len(self.pos_h) >= len(self.pos_h_set), "self.pos_h:{}, self.pos_h_set: {}".format(self.pos_h, self.pos_h_set)

        current_progress_ratio = traverse_length / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = is_overlap or is_hit_wall or traverse_length == max_markers

        reward = reward if self.env_task == 'oneStroke' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_doorkey_task_reward(self, agent_pos): ## doorkey
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        total_markers = np.sum(state[:,:,6:])
        error_markers = total_markers - 2
        score = 0
        if state[self.metadata['key'][0], self.metadata['key'][1], 5]: # [5, 2, 5]  # [1, 3, 7]
            score += 0.5
        if state[self.metadata['key'][0], self.metadata['key'][1], 5] and state[self.metadata['target'][0], self.metadata['target'][1], 7]:
            score += 0.5

        # open the door if marker picked
        if state[self.metadata['key'][0], self.metadata['key'][1], 5]:
            for door_pos in self.metadata['door_positions']: 
                self.s[door_pos[0], door_pos[1], 4] = False

        if error_markers > 0:
            score -= error_markers * 0.0001 # penalty

        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (current_progress_ratio==1.0) #or error_markers > 0 

        reward = reward if self.env_task == 'doorkey' else float(done)
        if self.env_task == 'doorkey_sparse':
            reward = reward if done and not self.done else 0
                
        self.done = self.done or done
        return reward, done

    def _get_seeder_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done

        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # calculate total_1 marker in the state
        existing_marker_num = len(self.metadata['existing_marker'])
        
        max_markers = (w-2)*(h-2) - existing_marker_num
       
        total_one_markers = np.sum(self.s[:,:,6])
        total_two_markers = np.sum(self.s[:,:,7])
        
        score = total_one_markers - existing_marker_num # - total_two_markers * 3

        current_progress_ratio = score / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = (total_one_markers == max_markers and total_two_markers == 0) or total_two_markers > 0

        reward = reward if self.env_task == 'seeder' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_snake_task_reward(self, agent_pos):
        # check if already done
        if self.done:
            return 0.0, self.done
 
        done = False
        w = self.w
        h = self.h
        state = self.s_h[-1]
        pos_tuple = tuple(agent_pos[:2])
        
        #is_hit_wall = self.a_h[-1] == 0 and pos_tuple == self.pos_h[-1]
        is_hit_body = self.s[agent_pos[0], agent_pos[1], 7]

        assert self.snake_len >= len(self.snake_body), "self.snake_len:{}, self.snake_body: {}".format(self.snake_len, self.snake_body)
        
        current_progress_ratio = (self.snake_len - 2) / 20.0 # max marker eatable: 10 (max snake length: 22)
        #current_progress_ratio = (2.0 ** ((self.snake_len - 2) / 4.0) -1) / 31.0 # max marker eatable: 5 (max snake length: 2 + 4*5 = 22)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio

        done = is_hit_body or current_progress_ratio >= 0.99 #is_hit_wall or current_progress_ratio >= 0.99

        reward = reward if self.env_task == 'snake' else float(done)
        self.done = self.done or done
        return reward, done


    def _get_state_reward(self, agent_pos, made_error=False):
        if self.env_task == 'cleanHouse' or self.env_task == 'cleanHouse_sparse':
            reward, done = self._get_cleanHouse_task_reward(agent_pos)
        elif self.env_task == 'harvester' or self.env_task == 'harvester_sparse':
            reward, done = self._get_harvester_task_reward(agent_pos)
        elif self.env_task == 'fourCorners' or self.env_task == 'fourCorners_sparse':
            reward, done = self._get_fourCorners_task_reward(agent_pos)
        elif self.env_task == 'randomMaze' or self.env_task == 'randomMaze_sparse':
            reward, done = self._get_randomMaze_task_reward(agent_pos)
        elif self.env_task == 'stairClimber' or self.env_task == 'stairClimber_sparse':
            reward, done = self._get_stairClimber_task_reward(agent_pos)
        elif self.env_task == 'topOff' or self.env_task == 'topOff_sparse':
            reward, done = self._get_topOff_task_reward(agent_pos)
        elif self.env_task == 'randomMaze_key2door' or self.env_task == 'randomMaze_key2door_sparse': 
            reward, done = self._get_randomMaze_key2door_task_reward(agent_pos)
        elif self.env_task == 'randomMaze_key2doorSpace' or self.env_task == 'randomMaze_key2doorSpace_sparse': 
            reward, done = self._get_randomMaze_key2doorSpace_task_reward(agent_pos)
        elif self.env_task == 'oneStroke' or self.env_task == 'oneStroke_sparse': ## oneStroke
            reward, done = self._get_oneStroke_task_reward(agent_pos)
        elif self.env_task == 'doorkey' or self.env_task == 'doorkey_sparse': ## doorkey
            reward, done = self._get_doorkey_task_reward(agent_pos)
        elif self.env_task == 'seeder' or self.env_task == 'seeder_sparse':
            reward, done = self._get_seeder_task_reward(agent_pos)
        elif self.env_task == 'snake' or self.env_task == 'snake_sparse':
            reward, done = self._get_snake_task_reward(agent_pos)
        else:
            raise NotImplementedError('{} task not yet supported'.format(self.env_task))

        return reward, done

    def print_state(self, state=None):
        agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        state = self.s_h[-1] if state is None else state
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:,:,4]] = 'x'
        state_2d[state[:,:,6]] = 'm'
        state_2d[state[:,:,7]] = 'M'
        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        for i in range(state_2d.shape[0]):
            print("".join(state_2d[i]))

    def render(self, mode='rgb_array'):
        return self.s_h[-1]

    # get location (x, y) and facing {north, east, south, west}
    def get_location(self):
        x, y, z = np.where(self.s[:, :, :4] > 0)
        return np.asarray([x[0], y[0], z[0]])

    # get the neighbor {front, left, right} location
    def get_neighbor(self, face):
        loc = self.get_location()
        if face == 'front':
            neighbor_loc = loc[:2] + {
                0: [-1, 0],
                1: [0, 1],
                2: [1, 0],
                3: [0, -1]
            }[loc[2]]
        elif face == 'left':
            neighbor_loc = loc[:2] + {
                0: [0, -1],
                1: [-1, 0],
                2: [0, 1],
                3: [1, 0]
            }[loc[2]]
        elif face == 'right':
            neighbor_loc = loc[:2] + {
                0: [0, 1],
                1: [1, 0],
                2: [0, -1],
                3: [-1, 0]
            }[loc[2]]
        return neighbor_loc

    ###################################
    ###    Perception Primitives    ###
    ###################################
    # return if the neighbor {front, left, right} of Karel is clear
    def neighbor_is_clear(self, face):
        self.perception_count += 1
        neighbor_loc = self.get_neighbor(face)
        if neighbor_loc[0] >= self.h or neighbor_loc[0] < 0 \
                or neighbor_loc[1] >= self.w or neighbor_loc[1] < 0:
            return False
        return not self.s[neighbor_loc[0], neighbor_loc[1], 4]

    def front_is_clear(self):
        return self.neighbor_is_clear('front')

    def left_is_clear(self):
        return self.neighbor_is_clear('left')

    def right_is_clear(self):
        return self.neighbor_is_clear('right')

    # return if there is a marker presented
    def marker_present(self):
        self.perception_count += 1
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) > 0

    def no_marker_present(self):
        self.perception_count += 1
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) == 0

    def get_perception_list(self):
        vec = ['frontIsClear', 'leftIsClear',
               'rightIsClear', 'markersPresent',
               'noMarkersPresent']
        return vec

    def get_perception_vector(self):
        vec = [self.front_is_clear(), self.left_is_clear(),
               self.right_is_clear(), self.marker_present(),
               self.no_marker_present()]
        return np.array(vec)

    ###################################
    ###       State Transition      ###
    ###################################
    # given a state and a action, return the next state
    def state_transition(self, a):
        made_error = False
        a_idx = np.argmax(a)
        loc = self.get_location()

        if a_idx == 0:
            # move
            if self.front_is_clear():
                front_loc = self.get_neighbor('front')
                loc_vec = self.s[loc[0], loc[1], :4]
                self.s[front_loc[0], front_loc[1], :4] = loc_vec
                self.s[loc[0], loc[1], :4] = np.zeros(4) > 0
                assert np.sum(self.s[front_loc[0], front_loc[1], :4]) > 0
 
                if self.env_task == "oneStroke" or self.env_task == "snake" and not self.done:
                    self.s[loc[0], loc[1], 7]  = True # change passed grid to double marker
                    self.s[loc[0], loc[1], 6]  = False
                    self.s[loc[0], loc[1], 5]  = False
                    
                    if self.env_task == "oneStroke":
                        self.s[loc[0], loc[1], 7]  = False
                        self.s[loc[0], loc[1], 4]  = True # change passed grid to wall
  
                if self.env_task == "snake" and not self.done:
                    if (front_loc[0], front_loc[1]) not in self.snake_body:
                        self.snake_body.append((loc[0], loc[1]))
                    else:
                        self.done = True
                        return
                    assert len(self.snake_body) >= 2
                    if self.s[front_loc[0], front_loc[1], 6] and self.snake_len < 22: # max snake length = 22
                        self.snake_len += 1 # += 2
                        self.s[front_loc[0], front_loc[1], 6] = False
                        # generate new marker
                        dummy_check = 0
                        while True:
                            #m_pos = np.random.randint(1, 7, size=[2])
                            #if (m_pos[0], m_pos[1]) != (loc[0], loc[1]) and np.sum(self.s[m_pos[0], m_pos[1], :5]) <= 0:
                            #    self.s[m_pos[0], m_pos[1], 6] = True
                            #    break
                            m_pos = self.snake_marker_list[self.snake_marker_pointer]
                            self.snake_marker_pointer = (self.snake_marker_pointer + 1) % len(self.snake_marker_list)
                            if np.sum(self.s[m_pos[0], m_pos[1], : ]) <= 0:
                                self.s[m_pos[0], m_pos[1], 6] = True
                                break
                            dummy_check += 1
                            if dummy_check > 50:
                                print("snake length: ", self.snake_len)
                                self.print_state()
                                assert False, "no valid marker found, m_pos: {}, state at m_pos:{}".format(m_pos, self.s[m_pos[0], m_pos[1]])
 
                    # check if snake tail should disappear
                    if len(self.snake_body) > self.snake_len:  
                        tail_pos = self.snake_body.popleft()
                        self.s[tail_pos[0], tail_pos[1], :] = np.zeros(len(state_table)) > 0

                assert np.sum(self.s[front_loc[0], front_loc[1], :4]) > 0
                next_loc = front_loc
            else:
                if self.make_error:
                    raise RuntimeError("Failed to move.")
                loc_vec = np.zeros(4) > 0
                loc_vec[(loc[2] + 2) % 4] = True  # Turn 180
                self.s[loc[0], loc[1], :4] = loc_vec
                next_loc = loc
            self.add_to_history(a_idx, next_loc)
        elif a_idx == 1 or a_idx == 2:
            # turn left or right
            loc_vec = np.zeros(4) > 0
            loc_vec[(a_idx * 2 - 3 + loc[2]) % 4] = True
            self.s[loc[0], loc[1], :4] = loc_vec
            self.add_to_history(a_idx, loc)

        elif a_idx == 3 or a_idx == 4:
            # pick up or put a marker
            num_marker = np.argmax(self.s[loc[0], loc[1], 5:])
            # just clip the num of markers for now
            if self.env_task in [
                    'topOff', 'topOff_sparse', 
                    'randomMaze_key2door', 'randomMaze_key2door_sparse', 
                    'randomMaze_key2doorSpace', 'randomMaze_key2doorSpace_sparse', 
                    'doorkey', 'doorkey_sparse', 
                    'seeder', 'seeder_sparse'
                    ]:
                new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER_topoff)
            else:
                new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER_general)
            #new_num_marker = a_idx*2-7 + num_marker
            #if new_num_marker < 0:
            #    if self.make_error:
            #        raise RuntimeError("No marker to pick up.")
            #    else:
            #        new_num_marker = num_marker
            #    made_error = True
            #elif new_num_marker > MAX_NUM_MARKER-1:
            #    if self.make_error:
            #        raise RuntimeError("Cannot put more marker.")
            #    else:
            #        new_num_marker = num_marker
            #    made_error = True
            marker_vec = np.zeros(MAX_NUM_MARKER_topoff+1) > 0
            marker_vec[new_num_marker] = True
            self.s[loc[0], loc[1], 5:] = marker_vec
            self.add_to_history(a_idx, loc, made_error)
        else:
            raise RuntimeError("Invalid action")
        return

    # given a karel env state, return a visulized image
    def state2image(self, s=None, grid_size=100, root_dir='./'):
        h = s.shape[0]
        w = s.shape[1]
        img = np.ones((h*grid_size, w*grid_size, 1))
        import pickle
        from PIL import Image
        import os.path as osp
        f = pickle.load(open(osp.join(root_dir, 'karel_env/asset/texture.pkl'), 'rb'))
        wall_img = f['wall'].astype('uint8')
        marker_img = f['marker'].astype('uint8')
        agent_0_img = f['agent_0'].astype('uint8')
        agent_1_img = f['agent_1'].astype('uint8')
        agent_2_img = f['agent_2'].astype('uint8')
        agent_3_img = f['agent_3'].astype('uint8')
        blank_img = f['blank'].astype('uint8')
        #blanks
        for y in range(h):
            for x in range(w):
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = blank_img
        # wall
        y, x = np.where(s[:, :, 4])
        for i in range(len(x)):
            img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = wall_img
        # marker
        y, x = np.where(np.sum(s[:, :, 6:], axis=-1))
        for i in range(len(x)):
            img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = marker_img
        # karel
        y, x = np.where(np.sum(s[:, :, :4], axis=-1))
        if len(y) == 1:
            y = y[0]
            x = x[0]
            idx = np.argmax(s[y, x])
            marker_present = np.sum(s[y, x, 6:]) > 0
            if marker_present:
                extra_marker_img = Image.fromarray(f['marker'].squeeze()).copy()
                if idx == 0:
                    extra_marker_img.paste(Image.fromarray(f['agent_0'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_0'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_0'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 1:
                    extra_marker_img.paste(Image.fromarray(f['agent_1'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_1'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_1'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 2:
                    extra_marker_img.paste(Image.fromarray(f['agent_2'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_2'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_2'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 3:
                    extra_marker_img.paste(Image.fromarray(f['agent_3'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_3'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_3'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
            else:
                if idx == 0:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_0']
                elif idx == 1:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_1']
                elif idx == 2:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_2']
                elif idx == 3:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_3']
        elif len(y) > 1:
            raise ValueError
        return img
