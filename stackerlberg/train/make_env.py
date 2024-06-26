import numpy as np
from ray.tune.registry import register_env as ray_register_env
from stackerlberg.envs.markov_game import MarkovGameEnv
from stackerlberg.envs.gsg_i import GSGEnv, make_args
from stackerlberg.envs.matrix_game import MatrixGameEnv, StochasticRewardWrapper
from stackerlberg.envs.test_envs import ThreadedTestEnv, ThreadedTestWrapper
from stackerlberg.wrappers.action_to_dist_wrapper import ActionToDistWrapper
from stackerlberg.wrappers.dict_to_discrete_obs_wrapper import DictToDiscreteObsWrapper
from stackerlberg.wrappers.learning_dynamics_wrapper import LearningDynamicsInfoWrapper
from stackerlberg.wrappers.observed_queries_wrapper import ObservedQueriesWrapper
from stackerlberg.wrappers.repeated_matrix_hypernetwork import (
    RepeatedMatrixHypernetworkWrapper,
)
from stackerlberg.wrappers.dict_to_image_obs_wrapper import DictToGridObsWrapper
from stackerlberg.wrappers.tabularq_wrapper import TabularQWrapper

registered_environments = {}


def register_env(name_or_function=None):
    """Decorator for registering an environment.
    Registeres the decorated function as a factory function for environments.
    Does this for both our own registry as well as rllib's."""
    if callable(name_or_function):
        # If we got a callable that means the decorator was called without paranthesis, i.e. @register
        # In that case we directly wrap the function
        n = name_or_function.__name__
        registered_environments[n] = name_or_function

        def env_creator_kwargs(env_config):
            return name_or_function(**env_config)

        ray_register_env(n, env_creator_kwargs)
        return name_or_function
    else:
        # Else we should have gotten a name string, so we return a decorator.
        def _register_env(function):

            if name_or_function is None:
                n = function.__name__
            else:
                n = name_or_function
            registered_environments[n] = function

            def env_creator_kwargs(env_config):
                return function(**env_config)

            ray_register_env(n, env_creator_kwargs)
            return function

        return _register_env


@register_env("test_env")
def make_test_env(
    num_agents: int = 2,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    env = ThreadedTestEnv(num_agents)
    env = ThreadedTestWrapper(env)
    return env


@register_env("matrix_game")
def make_matrix_env(
    episode_length: int = 1,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length)
    return env


@register_env("matrix_game_stackelberg_learning_dynamics")
def make_matrix_sld_env(
    n_follower_episodes: int = 32,
    n_reward_episodes: int = 4,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    mixed_strategies: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix)
    if mixed_strategies:
        env = ActionToDistWrapper(env)
    if not _is_eval_env:
        env = LearningDynamicsInfoWrapper(
            env, leader_agent_id="agent_0", n_follower_episodes=n_follower_episodes, n_reward_episodes=n_reward_episodes
        )
    return env


@register_env("matrix_game_stackelberg_observed_queries")
def make_matrix_observed_queries_env(
    n_samples: int = 1,
    samples_summarize: str = "list",
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    mixed_strategies: bool = False,
    reward_offset: float = 0.0,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, reward_offset=reward_offset)
    env = ObservedQueriesWrapper(
        env, leader_agent_id="agent_0", queries={"none": 0}, n_samples=n_samples, samples_summarize=samples_summarize
    )
    if mixed_strategies:
        env = ActionToDistWrapper(env)
    return env


@register_env("matrix_game_tabularq")
def make_matrix_tabularq_env(
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_hyper: bool = False,
    queries: bool = False,
    discrete_obs: bool = False,
    n_q_episodes: int = 50,
    q_alpha: float = 0.1,
    reset_between_episodes: bool = True,
    tell_leader: bool = False,
    tell_leader_mock: bool = False,
    hidden_queries: bool = False,
    follower_sparse_reward_prob: int = 1,
    follower_sparse_reward_scale: int = 1,
    follower_sparse_reward_deterministic: bool = False,
    leader_reward_during_q: bool = False,
    param_noise: bool = False,
    q_init_zero: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, reward_offset=0)
    if follower_sparse_reward_prob != 1:
        env = StochasticRewardWrapper(
            env,
            prob=follower_sparse_reward_prob,
            scale=follower_sparse_reward_scale,
            deterministic=follower_sparse_reward_deterministic,
        )
    env = TabularQWrapper(
        env,
        leader_agent_id="agent_0",
        follower_agent_id="agent_1",
        n_q_episodes=n_q_episodes,
        reset_between_episodes=reset_between_episodes,
        epsilon=0.1,
        alpha=q_alpha,
        tell_leader=tell_leader,
        tell_leader_mock=tell_leader_mock,
        hidden_queries=hidden_queries,
        param_noise=param_noise,
        leader_reward_during_q=leader_reward_during_q if not _is_eval_env else False,
        q_init_zero=q_init_zero,
    )
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env, leader_agent_id="agent_0", queries=queries, discrete=discrete_hyper)
    if discrete_obs and tell_leader:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_0")
    return env


@register_env("repeated_matrix_game")
def make_repeated_matrix_env(
    episode_length: int = 10,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_hyper: bool = False,
    queries: bool = False,
    discrete_obs: bool = False,
    small_memory: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length, memory=True, small_memory=small_memory)
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env, leader_agent_id="agent_0", queries=queries, discrete=discrete_hyper)
    if discrete_obs:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
    return env


@register_env("repeated_matrix_game_tabularq")
def make_repeated_matrix_tabularq_env(
    episode_length: int = 10,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_hyper: bool = False,
    queries: bool = False,
    discrete_obs: bool = False,
    small_memory: bool = False,
    n_q_episodes: int = 50,
    reset_between_episodes: bool = True,
    tell_leader: bool = False,
    tell_leader_mock: bool = False,
    hidden_queries: bool = False,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.9,
    q_init_zero: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == []:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length, memory=True, small_memory=small_memory)
    env = TabularQWrapper(
        env,
        leader_agent_id="agent_0",
        follower_agent_id="agent_1",
        n_q_episodes=n_q_episodes,
        reset_between_episodes=reset_between_episodes,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        tell_leader=tell_leader,
        tell_leader_mock=tell_leader_mock,
        hidden_queries=hidden_queries,
        q_init_zero=q_init_zero,
    )
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env, leader_agent_id="agent_0", queries=queries, discrete=discrete_hyper)
    if discrete_obs and tell_leader:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_0")
    return env

@register_env("markov_game_stackelberg_observed_queries")
def make_markov_observed_queries_env(
    episode_length: int = 10,
    n_samples: int = 1,
    samples_summarize: str = "list",
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_obs: bool = False,
    small_memory: bool = False,
    tell_leader: bool = False,
    tell_leader_mock: bool = False,
    hidden_queries: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    def generateQueryState(size=5, x1=None, y1=None, x2=None, y2=None):
        state = -np.ones((size, size), dtype=np.int)
        if x1 is None or y1 is None:
            x1, y1 = np.random.choice(size, 2)
            state[x1][y1] = 0
        else:
            state[x1][y1] = 0
        if x2 is None or y2 is None:
            x2, y2 = np.random.choice(size, 2)
            while state[x2][y2] == 0 :
                x2, y2 = np.random.choice(size, 2)
            state[x2][y2] = 1
        else:
            state[x2][y2] = 1
        return state

    env = MarkovGameEnv(episode_length=episode_length, memory=True, small_memory=small_memory, size=5)
    q0 = generateQueryState(size=3, x1=0, y1=0, x2=0, y2=2)
    q1 = generateQueryState(size=3, x1=0, y1=0, x2=2, y2=0)
    q2 = generateQueryState(size=3, x1=0, y1=0, x2=1, y2=1)
    q3 = generateQueryState(size=3, x1=0, y1=1, x2=0, y2=0)
    q4 = generateQueryState(size=3, x1=1, y1=0, x2=1, y2=1)
    q5 = generateQueryState(size=3, x1=2, y1=0, x2=2, y2=1)
    q6 = generateQueryState(size=3, x1=0, y1=1, x2=2, y2=0)
    q7 = generateQueryState(size=3, x1=1, y1=0, x2=0, y2=2)
    q8 = generateQueryState(size=3, x1=2, y1=1, x2=1, y2=1)
    if small_memory:

        qu = {}
    else:
        # q0 = generateQueryState(x1=4, y1=4, x2=0, y2=0)
        # q1 = generateQueryState(x1=3, y1=4, x2=1, y2=0)
        # q2 = generateQueryState(x1=4, y1=3, x2=0, y2=1)
        # q3 = generateQueryState(x1=3, y1=3, x2=1, y2=1)
        #
        # q4 = generateQueryState(x1=0, y1=0, x2=4, y2=4)
        # q5 = generateQueryState(x1=1, y1=0, x2=3, y2=4)
        # q6 = generateQueryState(x1=0, y1=1, x2=4, y2=3)
        # q7 = generateQueryState(x1=1, y1=1, x2=3, y2=3)
        #
        # q8 = generateQueryState(x1=1, y1=1, x2=2, y2=3)
        # q9 = generateQueryState(x1=1, y1=2, x2=2, y2=4)
        qu = {}
    env = ObservedQueriesWrapper(
        env,
        leader_agent_id="agent_0",
        queries=qu,
        n_samples=n_samples,
        samples_summarize=samples_summarize,
        tell_leader=tell_leader,
        tell_leader_mock=tell_leader_mock,
        hidden_queries=hidden_queries,
    )
    if discrete_obs:
        env = DictToGridObsWrapper(env, agent_id="agent_1")
        if tell_leader:
            env = DictToGridObsWrapper(env, agent_id="agent_0")
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env)
    return env

@register_env("gsg_stackelberg_observed_queries")
def make_gsg_observed_queries_env(
    episode_length: int = 100,
    n_samples: int = 1,
    samples_summarize: str = "list",
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_obs: bool = False,
    small_memory: bool = False,
    tell_leader: bool = False,
    tell_leader_mock: bool = False,
    hidden_queries: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    from stackerlberg.envs.maps import generate_map
    args = make_args()
    animal_density = generate_map(args)
    env = GSGEnv(args, animal_density=animal_density, episode_length=episode_length)
    qu = {}
    for i in range(100):
        qu["q_" + str(i)] = env.observation_space.sample()["agent_0"]

    env = ObservedQueriesWrapper(
        env,
        leader_agent_id="agent_0",
        queries=qu,
        n_samples=n_samples,
        samples_summarize=samples_summarize,
        tell_leader=tell_leader,
        tell_leader_mock=tell_leader_mock,
        hidden_queries=hidden_queries,
    )
    if discrete_obs:
        env = DictToGridObsWrapper(env, agent_id="agent_1")
        if tell_leader:
            env = DictToGridObsWrapper(env, agent_id="agent_0")
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env)
    return env

@register_env("repeated_matrix_game_stackelberg_observed_queries")
def make_repeated_matrix_observed_queries_env(
    episode_length: int = 10,
    n_samples: int = 1,
    samples_summarize: str = "list",
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    hypernetwork: bool = False,
    discrete_obs: bool = False,
    small_memory: bool = False,
    tell_leader: bool = False,
    tell_leader_mock: bool = False,
    hidden_queries: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == [] or matrix == () or len(matrix) == 0:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length, memory=True, small_memory=small_memory)
    if small_memory:
        qu = {"q0": 0, "q1": 1, "q2": 2}
    else:
        qu = {"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
    env = ObservedQueriesWrapper(
        env,
        leader_agent_id="agent_0",
        queries=qu,
        n_samples=n_samples,
        samples_summarize=samples_summarize,
        tell_leader=tell_leader,
        tell_leader_mock=tell_leader_mock,
        hidden_queries=hidden_queries,
    )
    if discrete_obs:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
        if tell_leader:
            env = DictToDiscreteObsWrapper(env, agent_id="agent_0")
    if hypernetwork:
        env = RepeatedMatrixHypernetworkWrapper(env)
    return env


@register_env("repeated_matrix_game_stackelberg_learning_dynamics")
def make_repeated_matrix_sld_env(
    episode_length: int = 10,
    n_follower_episodes: int = 1,
    n_reward_episodes: int = 1,
    matrix_name: str = "prisoners_dilemma",
    matrix: np.ndarray = [],
    mixed_strategies: bool = False,
    discrete_obs: bool = False,
    _deepmind: bool = True,
    _is_eval_env: bool = False,
    **kwargs,
):
    if matrix == [] or matrix == () or len(matrix) == 0:
        matrix = matrix_name
    env = MatrixGameEnv(matrix, episode_length=episode_length, memory=True)
    if mixed_strategies:
        env = ActionToDistWrapper(env)
    if not _is_eval_env:
        env = LearningDynamicsInfoWrapper(
            env, leader_agent_id="agent_0", n_follower_episodes=n_follower_episodes, n_reward_episodes=n_reward_episodes
        )
    if discrete_obs:
        env = DictToDiscreteObsWrapper(env, agent_id="agent_1")
    return env
