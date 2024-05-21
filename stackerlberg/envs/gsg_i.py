from copy import copy
from random import random
import random
import numpy as np
from gym import spaces
from gym.spaces import MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from stackerlberg.core.envs import MultiAgentWrapper
from tkinter import *
import argparse
def make_args():
    argparser = argparse.ArgumentParser()
    ########################################################################################
    ### Test parameters
    argparser.add_argument('--pa_load_path', type=str, default='./Results5x5/')
    argparser.add_argument('--po_load_path', type=str, default='./Results5x5/')
    argparser.add_argument('--load', type=bool, default=False)

    ### Environment
    argparser.add_argument('--row_num', type=int, default=5)
    argparser.add_argument('--column_num', type=int, default=5)
    argparser.add_argument('--ani_den_seed', type=int, default=66)
    argparser.add_argument('--max_time', type=int, default=4)

    ### Patroller
    argparser.add_argument('--pa_state_size', type=int, default=20)
    argparser.add_argument('--pa_num_actions', type=int, default=5)

    ### Poacher CNN
    argparser.add_argument('--snare_num', type=int, default=6)
    argparser.add_argument('--po_state_size', type=int, default=22)  # yf: add self footprint to poacher
    argparser.add_argument('--po_num_actions', type=int, default=10)

    ### Poacher Rule Base
    argparser.add_argument('--po_act_den_w', type=float, default=3.)
    argparser.add_argument('--po_act_enter_w', type=float, default=0.3)
    argparser.add_argument('--ipd_gsg_ppo_ppopo_act_leave_w', type=float, default=-1.0)
    argparser.add_argument('--po_act_temp', type=float, default=5.0)
    argparser.add_argument('--po_home_dir_w', type=float, default=3.0)
    argparser.add_argument('--experiment', type=str, default="ipd_gsg_ppo_ppo")

    ### Training
    argparser.add_argument('--map_type', type=str, default='random')
    argparser.add_argument('--advanced_training', type=bool, default=True)
    argparser.add_argument('--save_path', type=str, default='./Results33Parandom/')

    argparser.add_argument('--naive', type=bool, default=False)
    argparser.add_argument('--pa_episode_num', type=int, default=300000)
    argparser.add_argument('--po_episode_num', type=int, default=300000)
    argparser.add_argument('--pa_initial_lr', type=float, default=1e-4)
    argparser.add_argument('--po_initial_lr', type=float, default=5e-5)
    argparser.add_argument('--epi_num_incr', type=int, default=0)
    argparser.add_argument('--final_incr_iter', type=int, default=10)
    argparser.add_argument('--pa_replay_buffer_size', type=int, default=200000)
    argparser.add_argument('--po_replay_buffer_size', type=int, default=100000)
    argparser.add_argument('--test_episode_num', type=int, default=20000)
    argparser.add_argument('--iter_num', type=int, default=10)
    argparser.add_argument('--po_location', type=int, default=None)
    argparser.add_argument('--Delta', type=float, default=0.0)

    argparser.add_argument('--print_every', type=int, default=50)
    argparser.add_argument('--zero_sum', type=int, default=1)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--target_update_every', type=int, default=2000)
    argparser.add_argument('--reward_gamma', type=float, default=0.95)
    argparser.add_argument('--save_every_episode', type=int, default=5000)
    argparser.add_argument('--test_every_episode', type=int, default=2000)
    argparser.add_argument('--gui_every_episode', type=int, default=500)
    argparser.add_argument('--gui_test_num', type=int, default=20)
    argparser.add_argument('--gui', type=int, default=0)
    argparser.add_argument('--mix_every_episode', type=int, default=250)  # new added
    argparser.add_argument('--epsilon_decrease', type=float, default=0.05)  # new added
    argparser.add_argument('--reward_shaping', type=bool, default=False)
    argparser.add_argument('--PER', type=bool, default=False)
    #########################################################################################
    args = argparser.parse_args()
    return args
class GSGEnv(MultiAgentEnv):
    """A very basic marix game environment."""

    def __init__(
        self,
        args,
        animal_density,
        episode_length: int = 100,
        reward_offset: float = 0,
        **kwargs,
    ):
        """Creates a simple matrix game.
        Arguments:

        - matrix: A 3D numpy array of shape (rows, cols, 2) containing the payoff (bi-)matrix. Alternatively, a string can be passed, identifying one of several canonical games.
        - episode_length: The length of an episode.
        - memory: If True, agents can see the previous action of both agents."""
        super().__init__()
        self.num_agents = 2
        self._agent_ids = {"agent_0", "agent_1"}
        self.args = args
        self.row_num = self.args.row_num
        self.column_num = self.args.column_num
        self.poacher_first_home = True
        self.animal_density = animal_density
        self.home_flag = False
        self.catch_flag = False
        self.poacher_snare_num = args.snare_num
        self.canvas = None
        self.end_game = False
        self.action_space = spaces.Dict(
            {
                "agent_0": spaces.Discrete(5),
                "agent_1": spaces.Discrete(10),
            }
        )

        self.observation_space = spaces.Dict({
                "agent_0": spaces.Box(low=0, high=100.0, shape=(self.row_num, self.column_num, 20), dtype=np.float64),
                "agent_1": spaces.Box(low=0, high=100.0, shape=(self.row_num, self.column_num, 22), dtype=np.float64),
            })
        # self.observation_space = spaces.Dict({
        #     "agent_0":  spaces.Discrete(81),
        #     "agent_1":  spaces.Discrete(81),
        # })
        self.episode_length = episode_length
        self.current_step = 0
        self.reward_offset = reward_offset
        self.timestep = 0
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = ["agent_0", "agent_1"]

    def get_po_mode(self):
        if self.po_initial_loc == [0,0]:
            mode = '0'
        elif self.po_initial_loc == [0,self.column_num - 1]:
            mode = '1'
        elif self.po_initial_loc == [self.row_num - 1,0]:
            mode  = '2'
        elif self.po_initial_loc == [self.row_num - 1,self.column_num - 1]:
            mode = '3'
        return mode


    def reset(self):
        """Reset set the environment to a starting point.

               It needs to initialize the following attributes:
               - agents
               - timestamp
               - prisoner x and y coordinates
               - guard x and y coordinates
               - escape x and y coordinates
               - observation
               - infos

               And must set up the environment so that render(), step(), and observe() can be called without issues.
               """

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.current_step = 0

        self.poacher_snare_num = self.args.snare_num
        self.poacher_first_home = True
        self.home_flag = False
        self.catch_flag = False
        self.po_initial_loc = self.get_po_initial_loc(self.args.po_location)

        mode = np.random.randint(0,4)
        self.po_initial_loc = self.get_po_initial_loc(mode)
        self.po_loc = self.po_initial_loc
        self.pa_loc = self.pa_initial_loc = [self.row_num // 2, self.column_num // 2]
        self.pa_trace = {}
        self.po_trace = {}  # pos -> [(E, W, S, N)_0, (E, W, S, N)_1]
        self.snare_state = []
        self.snare_object = {}
        self.time = 0
        self.end_game = False
        self.pa_memory = np.zeros([self.row_num, self.column_num, 8])  # the memory of footprints
        self.po_memory = np.zeros([self.row_num, self.column_num, 8])
        self.po_self_memory = np.zeros(
            [self.row_num, self.column_num, 8])  # yf: add self footprint mem, [(E,S,W,N)_in, (E,S,W,N)_out]
        self.pa_self_memory = np.zeros(
            [self.row_num, self.column_num, 8])  # yf: add self footprint mem, [(E,S,W,N)_in, (E,S,W,N)_out]
        self.pa_visit_number = np.zeros([self.row_num, self.column_num])
        self.po_visit_number = np.zeros([self.row_num, self.column_num])

        for row in range(self.row_num):
            for col in range(self.column_num):
                self.pa_trace[(row, col)] = np.zeros(8)
                self.po_trace[(row, col)] = np.zeros(8)

        pa_obs = self.get_pa_state()
        po_obs = self.get_po_state()
        observations = {
            "agent_0": pa_obs,
            "agent_1": po_obs,
        }
        return observations

    def update_po_loc(self, action):
        if action == 'still':
            self.po_visit_number[self.po_loc[0], self.po_loc[1]] += 1
            return self.po_loc, self.po_loc

        if action == 'up':
            new_row, new_col = self.po_loc[0] - 1, self.po_loc[1]
            if self.in_bound(new_row, new_col):
                self.po_self_memory[new_row][new_col][1] = 1 # yf: add to self footprint memory
                self.po_self_memory[self.po_loc[0]][self.po_loc[1]][7] = 1
        elif action == 'down':
            new_row, new_col = self.po_loc[0] + 1, self.po_loc[1]
            if self.in_bound(new_row, new_col):
                self.po_self_memory[new_row][new_col][3] = 1
                self.po_self_memory[self.po_loc[0]][self.po_loc[1]][5] = 1
        elif action == 'left':
            new_row, new_col = self.po_loc[0], self.po_loc[1] - 1
            if self.in_bound(new_row, new_col):
                self.po_self_memory[new_row][new_col][0] = 1
                self.po_self_memory[self.po_loc[0]][self.po_loc[1]][6] = 1
        elif action == 'right':
            new_row, new_col = self.po_loc[0], self.po_loc[1] + 1
            if self.in_bound(new_row, new_col):
                self.po_self_memory[new_row][new_col][2] = 1
                self.po_self_memory[self.po_loc[0]][self.po_loc[1]][4] = 1
        else:
            print('Action Error!')
            exit(1)

        original_loc = self.po_loc
        if self.in_bound(new_row, new_col):
            self.po_loc = [new_row, new_col]

        if list(self.po_loc) != list(original_loc):  # in case the tuple and list issue
            self._update_po_trace(original_loc, self.po_loc, action)

        self.po_visit_number[self.po_loc[0], self.po_loc[1]] += 1
        return original_loc, self.po_loc

    def update_pa_loc(self, action):
        if action == 'still':
            self.pa_visit_number[self.pa_loc[0], self.pa_loc[1]] += 1
            return self.pa_loc, self.pa_loc

        if action == 'up':
            new_row, new_col = self.pa_loc[0] - 1, self.pa_loc[1]
            if self.in_bound(new_row, new_col):
                self.pa_self_memory[new_row][new_col][1] = 1 # yf: add to self footprint memory
                self.pa_self_memory[self.pa_loc[0]][self.pa_loc[1]][7] = 1
        elif action == 'down':
            new_row, new_col = self.pa_loc[0] + 1, self.pa_loc[1]
            if self.in_bound(new_row, new_col):
                self.pa_self_memory[new_row][new_col][3] = 1
                self.pa_self_memory[self.pa_loc[0]][self.pa_loc[1]][5] = 1
        elif action == 'left':
            new_row, new_col = self.pa_loc[0], self.pa_loc[1] - 1
            if self.in_bound(new_row, new_col):
                self.pa_self_memory[new_row][new_col][0] = 1
                self.pa_self_memory[self.pa_loc[0]][self.pa_loc[1]][6] = 1
        elif action == 'right':
            new_row, new_col = self.pa_loc[0], self.pa_loc[1] + 1
            if self.in_bound(new_row, new_col):
                self.pa_self_memory[new_row][new_col][2] = 1
                self.pa_self_memory[self.pa_loc[0]][self.pa_loc[1]][4] = 1
        else:
            print('Action Error!')
            exit(1)

        original_loc = self.pa_loc
        if self.in_bound(new_row, new_col):
            self.pa_loc = [new_row, new_col]

        if list(self.pa_loc) != list(original_loc):

            self._update_pa_trace(original_loc, self.pa_loc, action)

        self.pa_visit_number[self.pa_loc[0], self.pa_loc[1]] += 1
        return original_loc, self.pa_loc

    def map_action_to_string(self, action):
        action_str = {
            0: "still",
            1: "up",
            2: "down",
            3: "left",
            4: "right"
        }
        return action_str[action]

    def step(self, actions):
        # Execute actions
        po_action = actions["agent_1"]
        pa_action = actions["agent_0"]
        self.current_step += 1
        # Place the snare by the poacher
        snare_flag = False
        if po_action >= 5:
            po_action -= 5
            snare_flag = True

        po_action = self.map_action_to_string(po_action)
        pa_action = self.map_action_to_string(pa_action)
        if snare_flag and self.poacher_snare_num > 0:
            self.place_snare(self.po_loc)
            self.poacher_snare_num -= 1

        po_ori_loc, po_new_loc = self.update_po_loc(po_action)
        pa_ori_loc, pa_new_loc = self.update_pa_loc(pa_action)
        self.update_pa_memory()
        self.update_po_memory()

        if self.poacher_snare_num == 0 and tuple(self.po_initial_loc) == tuple(self.po_loc) and not tuple(self.pa_loc) == tuple(self.po_loc):
            self.home_flag = True
        pa_reward, po_reward = self.get_reward_train(po_ori_loc, po_new_loc, pa_ori_loc, pa_new_loc)
        if (self.catch_flag and len(self.snare_state) == 0) or (self.home_flag and len(self.snare_state) == 0):
            self.end_game = True
        else:
            self.end_game = False

        rewards = {
            "agent_0": pa_reward,
            "agent_1": po_reward,
        }
        pa_obs = self.get_pa_state()
        po_obs = self.get_po_state()
        observations = {
            "agent_0": pa_obs,
            "agent_1": po_obs,
        }
        finish = {"__all__": False}
        if self.end_game or self.current_step >= self.episode_length:
            finish = {"__all__": True }
            self.reset()

        return observations, rewards, finish, {}

    def _update_po_trace(self, ori_loc, new_loc, action):
        ori_loc = tuple(ori_loc)
        new_loc = tuple(new_loc)
        if action == 'up':
            self.po_trace[ori_loc][4] = 1
            self.po_trace[new_loc][1] = 1

        elif action == 'down':
            self.po_trace[ori_loc][5] = 1
            self.po_trace[new_loc][0] = 1

        elif action == 'left':
            self.po_trace[ori_loc][6] = 1
            self.po_trace[new_loc][3] = 1


        elif action == 'right':
            self.po_trace[ori_loc][7] = 1
            self.po_trace[new_loc][2] = 1
    def _update_pa_trace(self, ori_loc, new_loc, action):
        ori_loc = tuple(ori_loc)
        new_loc = tuple(new_loc)
        if action == 'up':
            self.pa_trace[ori_loc][4] = 1
            self.pa_trace[new_loc][1] = 1

        elif action == 'down':
            self.pa_trace[ori_loc][5] = 1
            self.pa_trace[new_loc][0] = 1


        elif action == 'left':
            self.pa_trace[ori_loc][6] = 1
            self.pa_trace[new_loc][3] = 1


        elif action == 'right':
            self.pa_trace[ori_loc][7] = 1
            self.pa_trace[new_loc][2] = 1
    def kill_animal(self, number = None):
        # Maybe this negative reward is not helpful.
        if number is None:
            kill_list = []
            for row, col in self.snare_state:
                if random.random() < (self.animal_density[row, col] / 5.):
                    kill_list.append([row, col])
            return kill_list
        else:
            kill_list = []
            # assert len(number) == len(self.snare_state)
            for i in range(len(self.snare_state)):
                if (1 << i) & number > 0:
                # if number[i] != 0:
                    kill_list.append([self.snare_state[i][0], self.snare_state[i][1]])
            return kill_list

    def get_reward_train(self, po_ori_loc, po_new_loc, pa_ori_loc, pa_new_loc, cfr=False):
        pa_reward, po_reward = 0.0, 0.0

        # from removing snares
        remove_num = 0
        if (self.pa_loc[0], self.pa_loc[1]) in self.snare_state:
            while (self.pa_loc[0], self.pa_loc[1]) in self.snare_state:
                pa_reward += 2
                remove_num += 1
                self.snare_state.remove((self.pa_loc[0], self.pa_loc[1]))



        # from killing animals
        kill_list = self.kill_animal()
        for row, col in kill_list:
            self.snare_state.remove((row, col))


        # poacher gets killing reward if it has not returned home, or has not been caught
        if not self.home_flag and not self.catch_flag:
            po_reward += 2 * len(kill_list)

        # from catch poachers
        # Only get catching reward when catch the poacher for the first time
        # only able to catch the poacher if it has not returned home
        if not self.catch_flag and not self.home_flag:
            if list(self.pa_loc) == list(self.po_loc):
                self.catch_flag = 1
                pa_reward += 8.
                po_reward -= 8.


        # Reward shaping
        po_ori_loc = np.array(po_ori_loc)
        po_new_loc = np.array(po_new_loc)
        pa_ori_loc = np.array(pa_ori_loc)
        pa_new_loc = np.array(pa_new_loc)
        po_initial_loc = np.array(self.po_initial_loc)

        if self.args.reward_shaping:
            # patroller reward shaping: the distance change to the poacher
            if not self.catch_flag and not self.home_flag:
                pa_reward += np.sqrt(np.sum(np.square(pa_ori_loc - po_ori_loc))) - \
                             np.sqrt(np.sum(np.square(pa_new_loc - po_new_loc)))

            # patroller reward shaping: weighted distance change to the snares
            if len(self.snare_state) > 0:
                snare_pos = np.array(self.snare_state)
                snare_weight = np.array([self.animal_density[i, j] for i, j in self.snare_state]) / 0.6

                pa_reward += np.mean(
                    snare_weight * (
                            np.sqrt(np.sum(np.square(pa_ori_loc - snare_pos), axis=1)) -
                            np.sqrt(np.sum(np.square(pa_new_loc - snare_pos), axis=1))
                    )
                )

        # Go away from patroller
        if not self.catch_flag and not self.home_flag:
            far_pa_reward = np.sqrt(np.sum(np.square(pa_new_loc - po_new_loc))) - \
                            np.sqrt(np.sum(np.square(pa_ori_loc - po_ori_loc)))
            if far_pa_reward < 0:
                po_reward += far_pa_reward

        # poacher reward shaping: get weighted reward from putted snares
        if not self.catch_flag and not self.home_flag:
            if len(self.snare_state) > 0:
                snare_weight = np.array([self.animal_density[i, j] for i, j in self.snare_state]) * 0.6
                po_reward += np.sum(snare_weight)

        # poacher reward shaping: encourage to go back home when snares are run out
        # if not self.home_flag and no_snare and not self.catch_flag:
        #     po_reward += np.sqrt(np.sum(np.square(po_ori_loc - po_initial_loc))) - \
        #                     np.sqrt(np.sum(np.square(po_new_loc - po_initial_loc)))

        # poacher reward shaping: when first get back to home, get a positive reward
        # if self.home_flag and not self.catch_flag:
        #     if self.poacher_first_home:
        #         po_reward += 0.5
        #     self.poacher_first_home = False
        if cfr:
            return pa_reward, po_reward, remove_num
        return pa_reward, po_reward

    def get_reward_test(self, cfr=False, number=None):

        pa_reward, po_reward = 0.0, 0.0
        remove_cnt = 0

        ### Patroller will clear the snare if find one
        if (self.pa_loc[0], self.pa_loc[1]) in self.snare_state:
            while (self.pa_loc[0], self.pa_loc[1]) in self.snare_state:
                remove_cnt += 1
                pa_reward += 2
                self.snare_state.remove((self.pa_loc[0], self.pa_loc[1]))


        ### kill animals
        kill_list = self.kill_animal(number=number)
        if not self.catch_flag and not self.home_flag:
            po_reward += 2 * len(kill_list)
        pa_reward -= 2 * len(kill_list)
        for row, col in kill_list:
            self.snare_state.remove((row, col))


        # only get this negative reward of being caught for the first time
        if not self.catch_flag and not self.home_flag:
            # If get caught
            if list(self.pa_loc) == list(self.po_loc):
                self.catch_flag = 1
                po_reward -= 8
                pa_reward += 8


        if self.args.zero_sum == 0:
            return pa_reward, po_reward
        else:
            if not cfr:
                return pa_reward, -pa_reward
            else:
                return pa_reward, -pa_reward, remove_cnt

        # return pa_reward, -pa_reward

    def place_snare(self, loc):
        self.snare_state.append((loc[0], loc[1]))


    def get_local_ani_den(self, loc):
        if self.in_bound(loc[0], loc[1]):
            den = [self.animal_density[loc[0], loc[1]]]
        else:
            den = [0.]

        up_loc = (loc[0] - 1, loc[1])
        if self.in_bound(up_loc[0], up_loc[1]):
            den.append(self.animal_density[up_loc[0], up_loc[1]])
        else:
            den.append(0.)

        down_loc = (loc[0] + 1, loc[1])
        if self.in_bound(down_loc[0], down_loc[1]):
            den.append(self.animal_density[down_loc[0], down_loc[1]])
        else:
            den.append(0.)

        left_loc = (loc[0], loc[1] - 1)
        if self.in_bound(left_loc[0], left_loc[1]):
            den.append(self.animal_density[left_loc[0], left_loc[1]])
        else:
            den.append(0.)

        right_loc = (loc[0], loc[1] + 1)
        if self.in_bound(right_loc[0], right_loc[1]):
            den.append(self.animal_density[right_loc[0], right_loc[1]])
        else:
            den.append(0.)

        return np.array(den)

    def get_local_pa_trace(self, loc):
        if tuple(loc) in self.pa_trace:
            return self.pa_trace[tuple(loc)]
        else:
            return np.zeros(8)

    def get_local_po_trace(self, loc):
        return self.po_trace[tuple(loc)]

    def update_pa_memory(self):
        self.pa_memory[self.pa_loc[0], self.pa_loc[1]] = self.po_trace[tuple(self.pa_loc)]

    def update_po_memory(self):
        self.po_memory[self.po_loc[0], self.po_loc[1]] = self.pa_trace[tuple(self.po_loc)]

    def get_pa_state(self):
        state = self.pa_memory

        # yf: add self footprint memory as state
        state = np.concatenate((state, self.pa_self_memory), axis=2)

        ani_den = np.expand_dims(self.animal_density, axis=2)
        state = np.concatenate((state, ani_den), axis=2)

        coordinate = np.zeros([self.row_num, self.column_num])
        coordinate[self.pa_loc[0], self.pa_loc[1]] = 1
        coordinate = np.expand_dims(coordinate, axis=2)
        state = np.concatenate((state, coordinate), axis=2)

        visit_num_norm = np.expand_dims(self.pa_visit_number / 10., axis=2)
        state = np.concatenate((state, visit_num_norm), axis=2)

        time_left = np.ones([self.row_num, self.column_num]) * float(self.time) / (self.args.max_time / 2.)
        time_left = np.expand_dims(time_left, axis=2)
        state = np.concatenate((state, time_left), axis=2)
        assert state.shape == (self.row_num, self.column_num, self.args.pa_state_size)
        return state

    def get_po_state(self):
        snare_num = self.poacher_snare_num
        state = self.po_memory

        # yf: add self footprint memory as state
        state = np.concatenate((state, self.po_self_memory), axis=2)

        ani_den = np.expand_dims(self.animal_density, axis=2)
        state = np.concatenate((state, ani_den), axis=2)

        coordinate = np.zeros([self.row_num, self.column_num])
        coordinate[self.po_loc[0], self.po_loc[1]] = 1.
        coordinate = np.expand_dims(coordinate, axis=2)
        state = np.concatenate((state, coordinate), axis=2)

        visit_num_norm = np.expand_dims(self.po_visit_number / 10., axis=2)
        state = np.concatenate((state, visit_num_norm), axis=2)

        snare_num_left = np.ones([self.row_num, self.column_num]) * float(snare_num) / self.args.snare_num
        snare_num_left = np.expand_dims(snare_num_left, axis=2)
        state = np.concatenate((state, snare_num_left), axis=2)

        time_left = np.ones([self.row_num, self.column_num]) * float(self.time) / (self.args.max_time / 2.)
        time_left = np.expand_dims(time_left, axis=2)
        state = np.concatenate((state, time_left), axis=2)

        initial_loc = np.zeros([self.row_num, self.column_num])
        initial_loc[self.po_initial_loc[0], self.po_initial_loc[1]] = 1.
        initial_loc = np.expand_dims(initial_loc, axis=2)
        state = np.concatenate((state, initial_loc), axis=2)
        assert state.shape == (self.row_num, self.column_num, self.args.po_state_size)
        return state

    def get_local_snare(self, loc):
        if (loc[0], loc[1]) in self.snare_state:
            num = 0
            for snare_loc in self.snare_state:
                if snare_loc == (loc[0], loc[1]):
                    num += 1
            return num
        else:
            return 0.

    def in_bound(self, row, col):
        return row >= 0 and row <= (self.row_num - 1) and col >= 0 and col <= (self.column_num - 1)

    def get_po_initial_loc(self, idx=None):
        candidate = [[0, 0], [0, self.column_num - 1], [self.row_num - 1, self.column_num - 1], [self.row_num - 1, 0]]

        if idx is not None:
            return candidate[idx]

        index = random.randint(0, 3)
        return candidate[index]


class StochasticRewardWrapper(MultiAgentWrapper):
    """Makes reward stochastich and sparser, but with same expectation."""

    def __init__(self, env, prob: float = 1, scale: float = 1, agent: str = "agent_1", deterministic: bool = False, **kwargs):
        super().__init__(env, **kwargs)
        self.scale = scale
        self.prob = prob
        self.agent = agent
        self.deterministic = deterministic

    def reset(self):
        self._step_counter = 0
        return self.env.reset()

    def step(self, actions):
        self._step_counter += 1
        obs, rewards, dones, infos = self.env.step(actions)
        if self.agent in rewards:
            if self.deterministic:
                rewards[self.agent] = rewards[self.agent] * self.scale if self._step_counter >= self.prob else 0
            else:
                rewards[self.agent] *= self.scale * np.random.binomial(1, 1 / self.prob)
        return obs, rewards, dones, infos
