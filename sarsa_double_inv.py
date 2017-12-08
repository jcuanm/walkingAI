from sarsa_inv_pen import SarsaInvPend


class SarsaDoubleInvPend(SarsaInvPend):
    ## Learning related constants
    INIT_EXP = 0.7
    MIN_EXPLORATION = 1e-10
    EXP_COOLDOWN = 15

    INIT_LEARN = 0.8
    MIN_LEARNING = 1e-8
    LEARN_COOLDOWN = 20

    DISCOUNT = 0.99

    ## Discretization related constants
    # number of discrete states
    NUM_BUCKETS = (20, 20, 20, 1, 1, 20, 20, 20, 1, 1, 1)  # x, sin, sin, cos, cos, v, w, w, , ,

    # number of action states
    OFFSET = 3
    SCALE = 3
    NUM_ACTIONS = 2 * OFFSET + 1
    ACTION_CONSTRAINT = 0.1

    def __init__(self, env, explore=INIT_EXP, learn=INIT_LEARN):
        SarsaInvPend.__init__(self, env, explore, learn)


    def discretize(self):
        # Bounds for each discrete state
        state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        state_bounds = [(-1, 1) for _ in state_bounds]
        state_bounds[3] = 0.5, 2
        state_bounds[4] = 0.5, 2
        state_bounds[6] = -5, 5
        state_bounds[7] = -10, 10
        return state_bounds

    def reward(self, obs, prev_obs, default_reward):
        #reward = - 0.1*abs(obs[1] - obs[2])**2 - 0.1*abs(obs[6] - obs[7])**2
        reward = abs((1-abs(obs[1]))/(obs[6]-prev_obs[6])) + abs((1-abs(obs[2]))/(obs[7]-prev_obs[7]))
        #print(reward)
        return reward
