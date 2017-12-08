import gym
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from mujoco_inv_pend import MujocoInvPend

class Walker(MujocoInvPend):
    # Learning related constants
    INIT_EXP = 0.8
    MIN_EXPLORATION = 1e-8
    EXP_COOLDOWN = 1000

    INIT_LEARN = 0.7
    MIN_LEARNING = 1e-7
    LEARN_COOLDOWN = 1000

    DISCOUNT = 0.99

    # Discretization related constants
    # Number of discrete states (bucket) per state dimension
    NUM_BUCKETS = (1, 10, 10, 10, 10, 1, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    # number of action states
    OFFSET = 5
    SCALE = 5
    NUM_ACTIONS = 2 * OFFSET + 1
    ACTION_CONSTRAINT = 0.05

    def __init__(self, env, explore=INIT_EXP, learn=INIT_LEARN):
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

    def act(self, observation):
        self.action = 0,0,0,0,0,0
        return self.action

    def reward(self, obs, prev_obs, default_reward):
        return default_reward

    def step(self, action):
        """Compute environment at next time step"""
        self.action = action
        return self.env.step(action)
