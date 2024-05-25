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

        self.dim_box = 0
        self.dim_action = 0
        for obs_type, space in env.observation_space[agent_id].spaces.items():
            if isinstance(space, spaces.Box):
                self.dim_box = space.shape
            elif isinstance(space, spaces.Discrete):
                self.dim_action += 1

        self.observation_space[agent_id] = spaces.Dict({
            "original_obs": spaces.Box(low=0, high=100.0, shape=self.dim_box, dtype=np.float64),
            "observed_action": spaces.Box(low=0, high=5, shape=(self.dim_action,), dtype=np.float64)
        })

    def encode_obs(self, obs, agent_id):
        flatten_action = {}
        observed_action = []
        for obs_type, obs_value in obs.items():
            if len(obs_value.shape) > 1:
                flatten_action["original_obs"] = obs_value
            else:
                observed_action.append(obs_value)
        observed_action = np.array(observed_action, dtype=np.float64)
        flatten_action["observed_action"] = observed_action
        return flatten_action

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
