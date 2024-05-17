import copy
from math import isnan
from typing import Optional

import gym
import numpy as np
from gym import spaces

from stackerlberg.core.envs import MultiAgentEnv, MultiAgentWrapper


class DictToGridObsWrapper(MultiAgentWrapper):
    """This wrapper converts a Dict(Discrete) observation space into a Discrete observation space."""

    def __init__(self, env: MultiAgentEnv, agent_id: str = "agent_1", indentifier: Optional[str] = None):
        super().__init__(env, indentifier)
        self.agent_id = agent_id
        self.observation_space = copy.deepcopy(env.observation_space)
        self.action_space = copy.deepcopy(env.action_space)
        assert isinstance(env.observation_space[agent_id], spaces.Dict), "DictToDiscreteObsWrapper only works with Dict observation spaces"

        dim = 0
        for obs_type, space in env.observation_space[agent_id].spaces.items():
            if isinstance(space, spaces.Box):
                dim += space.shape[0] * space.shape[1]
            elif isinstance(space, spaces.Discrete):
                dim += space.n

        self.observation_space[agent_id] = spaces.Box(low=-1, high=1, shape=(dim, ), dtype=np.int)

    def encode_obs(self, obs, agent_id):
        flatten_obs = []
        for obs_type, obs_value in obs.items():
            if isinstance(obs_value, np.ndarray):
                flatten_obs.append(obs_value.reshape(-1))
            else:
                actions = np.ones(self.action_space[agent_id].n)
                actions[obs_value] = 1
                flatten_obs.append(actions)
        flatten_obs = np.concatenate(flatten_obs)
        return flatten_obs

    def reset(self):
        observation = self.env.reset()
        if self.agent_id in observation:
            agent_obs = observation[self.agent_id]  # this is a dict.
            observation[self.agent_id] = self.encode_obs(agent_obs, self.agent_id)
        return observation

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions)
        if self.agent_id in observation:
            agent_obs = observation[self.agent_id]  # this is a dict.
            observation[self.agent_id] = self.encode_obs(agent_obs, self.agent_id)
        return observation, reward, done, info
