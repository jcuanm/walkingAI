import numpy as np
import random
import math


class MujocoInvPend(object):
    ## Learning related constants
    INIT_EXP = 0.5
    MIN_EXPLORATION = 1e-10
    EXP_COOLDOWN = 15

    INIT_LEARN = 0.5
    MIN_LEARNING = 1e-9
    LEARN_COOLDOWN = 20

    DISCOUNT = 0.99

    ## Discretization related constants
    # number of discrete states
    NUM_BUCKETS = (1, 40, 1, 20)  # x, theta, x', theta'

    # number of action states
    OFFSET = 3
    SCALE = 1
    NUM_ACTIONS = 2*OFFSET+1
    ACTION_CONSTRAINT = 0.1

    def __init__(self, env, explore=INIT_EXP, learn=INIT_LEARN):
        self.env = env
        self.exploration_rate = explore
        self.learning_rate = learn
        self.q_table = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS,))  # Q-Table for each state-action pair
        self.state_bounds = self.discretize()

        self.prev_state = ()
        self.state = ()
        self.action = 0

    def discretize(self):
        """Discretize state space into buckets"""
        # Bounds for each discrete state
        state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        state_bounds = [(-3, 3) for _ in state_bounds]
        state_bounds[1] = -2, 2
        state_bounds[3] = -math.radians(80), math.radians(80)
        return state_bounds

    def act(self, observation):
        """Follow epsilon-greedy policy to compute next action"""
        self.prev_state = self.state_to_bucket(observation)

        # Select a random action
        if random.random() < self.exploration_rate:
            self.action = int(self.env.action_space.sample()*self.SCALE) + self.OFFSET
        # Select the action with the highest q
        else:
            self.action = np.argmax(self.q_table[self.prev_state])
        return self.action

    def step(self, action):
        """Compute environment at next time step"""
        self.action = action
        return self.env.step((action - self.OFFSET) * self.ACTION_CONSTRAINT)

    def reward(self, obs, prev_obs, default_reward):
        """Reward function"""
        reward = 1.0 - 5*abs(obs[1] + obs[3])
        return reward

    def update(self, obs, reward):
        """Update Q value table"""
        self.state = self.state_to_bucket(obs)

        # Update the Q based on the result
        best_q = np.amax(self.q_table[self.state])
        self.q_table[self.prev_state + (self.action,)] += self.learning_rate * (reward + self.DISCOUNT * best_q - self.q_table[self.prev_state + (self.action,)])

        self.prev_state = self.state

    def state_to_bucket(self, obs):
        """Divides continuous state space into desired number of discrete buckets"""
        bucket_indices = []
        # print(self.obs)
        for i in range(len(obs)):
            if obs[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif obs[i] >= self.state_bounds[i][1]:
                bucket_index = self.NUM_BUCKETS[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = (self.NUM_BUCKETS[i] - 1) * self.state_bounds[i][0] / bound_width
                scaling = (self.NUM_BUCKETS[i] - 1) / bound_width
                bucket_index = int(round(scaling * obs[i] - offset))
            bucket_indices.append(bucket_index)
        return tuple(bucket_indices)

    def get_explore_rate(self, episode):
        """Cool down exploration rate logarithmically"""
        self.exploration_rate = max(self.MIN_EXPLORATION, min(self.INIT_EXP, 1.0 - math.log10((episode + 1) / self.EXP_COOLDOWN)))

    def get_learning_rate(self, episode):
        """Cool down learning rate logarithmically"""
        self.learning_rate = max(self.MIN_LEARNING, min(self.INIT_LEARN, 1.0 - math.log10((episode + 1) / self.LEARN_COOLDOWN)))


