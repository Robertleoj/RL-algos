import gymnasium as gym

def env_is_discrete(env):
    return isinstance(env.action_space, gym.spaces.Discrete)

def env_info(env):
    """
    output: is_discrete, obs_space, action_space, action_bounds
    """
    obs_space = env.observation_space.shape[0]

    action_bounds = None

    is_discrete = env_is_discrete(env)

    # if discrete, use a multinomial policy net
    if is_discrete:
        # get the number of actions
        action_space = env.action_space.n

    else:
        # get the number of actions
        action_space = env.action_space.shape[0]

        # get the action space bounds, low and high
        action_space_low = env.action_space.low
        action_space_high = env.action_space.high

        action_bounds = list(zip(action_space_low, action_space_high))


    return is_discrete, obs_space, action_space, action_bounds
