from mujoco_inverted_pendulum import MujocoInvPend


class DoubleInvPend(MujocoInvPend):
    # Learning related constants
    INIT_EXP = 0.8
    MIN_EXPLORATION = 1e-8
    EXP_COOLDOWN = 1000

    INIT_LEARN = 0.7
    MIN_LEARNING = 1e-7
    LEARN_COOLDOWN = 1000

    DISCOUNT = 0.99

    # Discretization related constants
    # number of discrete states
    NUM_BUCKETS = (1, 10, 20, 1, 1, 1, 10, 20, 1, 1, 1)  # x, sin, sin, cos, cos, v, w, w, , ,

    # number of action states
    OFFSET = 5
    SCALE = 5
    NUM_ACTIONS = 2 * OFFSET + 1
    ACTION_CONSTRAINT = 0.05

    def __init__(self, env, explore=INIT_EXP, learn = INIT_LEARN):
        MujocoInvPend.__init__(self, env, explore, learn)

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
        #reward = abs((1-abs(obs[1]))-(obs[6]-prev_obs[6])) + abs((1-abs(obs[2]))-(obs[7]-prev_obs[7])) - obs[1]*obs[6] \
                 # - obs[2]*obs[7] - abs(obs[6]) - abs(obs[7]) - abs(obs[1]) - abs(obs[2]) #figure 2
        reward = 2*abs((1 - abs(obs[1])) - 2*(obs[6] - prev_obs[6])) + abs((1 - abs(obs[2])) - (obs[7] - prev_obs[7])) - \
                 2*obs[1] * obs[6] - obs[2] * obs[7] - abs(obs[6]) - abs(obs[7])

        #reward = 5.0 - obs[1]**2 - obs[2]**2 - 0.1*obs[6]**2 - 0.1*obs[7]**2 \
        #         - 0.001*((self.action - self.OFFSET) * self.ACTION_CONSTRAINT)**2
        #print(obs)
        #print(reward)
        return reward


