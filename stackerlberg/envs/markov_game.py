from copy import copy
from random import random

import numpy as np
from gym import spaces
from gym.spaces import MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from stackerlberg.core.envs import MultiAgentWrapper


class MarkovGameEnv(MultiAgentEnv):
    """A very basic marix game environment."""

    def __init__(
        self,
        matrix="prisoner_guard",
        episode_length: int = 1,
        memory: bool = False,
        small_memory: bool = False,
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
        self.action_space = spaces.Dict(
            {
                "agent_0": spaces.Discrete(4),
                "agent_1": spaces.Discrete(4),
            }
        )
        self.memory = memory
        self.small_memory = small_memory
        self.observation_space = spaces.Dict({
                "agent_0": spaces.Box(low=-1.0, high=1.0, shape=(5, 5), dtype=np.int),
                "agent_1": spaces.Box(low=-1.0, high=1.0, shape=(5, 5), dtype=np.int),
            })
        # self.observation_space = spaces.Dict({
        #     "agent_0":  spaces.Discrete(81),
        #     "agent_1":  spaces.Discrete(81),
        # })
        self.episode_length = episode_length
        self.current_step = 0
        self.reward_offset = reward_offset
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = ["agent_0", "agent_1"]

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

        state = -np.ones((5, 5), dtype=np.int)
        self.guard_x, self.guard_y = np.random.choice(5, 2)
        state[self.guard_x][self.guard_y] = 0

        self.prisoner_x, self.prisoner_y = np.random.choice(5, 2)
        while state[self.prisoner_x][self.prisoner_y] == 0:
            self.prisoner_x, self.prisoner_y = np.random.choice(5, 2)
        state[self.prisoner_x][self.prisoner_y] = 1


        self.escape_x = 2
        self.escape_y = 2

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: state for a in self.agents}

        return infos
    def step(self, actions):
        # Execute actions
        prisoner_action = actions["agent_1"]
        guard_action = actions["agent_0"]
        self.current_step += 1
        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 4:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 4:
            self.prisoner_y += 1

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < 4:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < 4:
            self.guard_y += 1

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"agent_1": -10, "agent_0": 10}
            terminations = {a: True for a in self.agents}

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"agent_1": 10, "agent_0": -10}
            terminations = {a: True for a in self.agents}
        else:
            rewards = {"agent_1": -1, "agent_0": -1}
        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep == self.episode_length - 1:
            rewards = {"agent_1": 0, "agent_0": 0}
            truncations = {"agent_1": True, "agent_0": True}
        self.timestep += 1

        obs = -np.ones((5, 5))
        obs[self.guard_x][self.guard_y] = 0
        obs[self.prisoner_x][self.prisoner_y] = 1
        # Get observations
        observations = {
            "agent_1": obs,
            "agent_0": obs
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}
        finish = {"__all__": False}
        if any(terminations.values()) or all(truncations.values()):
            finish = {"__all__": True }
            self.reset()

        return observations, rewards, finish, {}
    # def step(self, actions):
    #
    #     if self.memory is False:
    #         obs = {"agent_0": 0, "agent_1": 0}
    #     else:
    #         if self.small_memory is False:
    #             obs = {
    #                 "agent_0": 1 + actions["agent_0"] + 2 * actions["agent_1"],
    #                 "agent_1": 1 + actions["agent_0"] + 2 * actions["agent_1"],
    #             }
    #             # 0 : first step, 1 (0,0), 2 (1,0), 3 (0,1), 4 (1,1)
    #             # 1, 2: agent 1 action 0, 3, 4 action 1
    #             # 1, 3: agent 0 action 0, 2, 4 action 1
    #         else:
    #             obs = {"agent_0": 1 + actions["agent_1"], "agent_1": 1 + actions["agent_0"]}
    #             # 0: first step
    #             # 1: other agent cooperated
    #             # 2: other agent defected
    #     return obs, rewards, {"__all__": True if self.current_step >= self.episode_length else False}, {}


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
