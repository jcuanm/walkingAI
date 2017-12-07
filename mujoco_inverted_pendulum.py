import gym
import numpy as np
import random
import math
from matplotlib import pyplot as plt


class MujocoInvPend(object):
    ## Learning related constants
    INIT_EXP = 1.0
    MIN_EXPLORATION = 1e-10
    EXP_COOLDOWN = 15

    INIT_LEARN = 0.5
    MIN_LEARNING = 1e-8
    LEARN_COOLDOWN = 20

    DISCOUNT = 0.99

    ## Discretization related constants
    # number of discrete states
    NUM_BUCKETS = (1, 40, 1, 20)  # x, theta, x', theta'
    # number of action states
    NUM_ACTIONS = 7

    def __init__(self, env):
        self.env = env
        self.exploration_rate = 1.0
        self.learning_rate = 0.5
        self.q_table = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS,)) # Q-Table for each state-action pair
        self.state_bounds = self.discretize()

        self.prev_state = ()
        self.state = ()
        self.action = 0

    def discretize(self):
        # Bounds for each discrete state
        state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        state_bounds = [(-3, 3) for _ in state_bounds]
        state_bounds[1] = -2, 2
        state_bounds[3] = -math.radians(80), math.radians(80)
        return state_bounds

    def act(self, observation):
        self.prev_state = self.state_to_bucket(observation)

        # Select a random action
        if random.random() < self.exploration_rate:
            self.action = int(self.env.action_space.sample()) + 3
        # Select the action with the highest q
        else:
            self.action = np.argmax(self.q_table[self.prev_state])
        return self.action

    def step(self, action):
        self.action = action
        return self.env.step((action - 3) * 0.1)

    def reward(self, obs):
        return 1.0

    def update(self, obs, reward):
        self.state = self.state_to_bucket(obs)

        # Update the Q based on the result
        best_q = np.amax(self.q_table[self.state])
        self.q_table[self.prev_state + (self.action,)] += self.learning_rate * (reward + self.DISCOUNT * best_q - self.q_table[self.prev_state + (self.action,)])

        self.prev_state = self.state

    def state_to_bucket(self, obs):
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
        self.exploration_rate = max(MIN_EXPLORATION, min(self.INIT_EXP, 1.0 - math.log10((episode + 1) / self.EXP_COOLDOWN)))

    def get_learning_rate(self, episode):
        self.learning_rate = max(MIN_LEARNING, min(self.INIT_LEARN, 1.0 - math.log10((episode + 1) / self.LEARN_COOLDOWN)))






## Episode related constants
NUM_EPISODES = 500
TIME_STEPS = 500

def init(name):
    env = gym.make(name)
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    return env, action_dim, observation_dim

## Learning related constants
MIN_EXPLORATION = 1e-10
MIN_LEARNING = 1e-8
DEBUG_MODE = False

def main():
    env, action_dim, observation_dim = init('InvertedPendulum-v1')

    # Number of discrete states (bucket) per state dimension
    NUM_BUCKETS = (1, 40, 1, 20) # x, theta, x', theta'

    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    STATE_BOUNDS = [(-3, 3) for item in STATE_BOUNDS]
    STATE_BOUNDS[1] = -2, 2
    STATE_BOUNDS[3] = -math.radians(80), math.radians(80)

    ## Creating a Q-Table for each state-action pair
    q_table = np.zeros(NUM_BUCKETS + (7,))

    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    times = []

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv, STATE_BOUNDS, NUM_BUCKETS)

        for t in range(TIME_STEPS):
            #if episode > 499:
            #    env.render()

            # Select an action
            action = select_action(env, state_0, explore_rate, q_table)

            # Execute the action
            obv, reward, done, _ = env.step((action - 3)*0.1)
            
            # Observe the result
            state = state_to_bucket(obv, STATE_BOUNDS, NUM_BUCKETS)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*best_q - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            if done:
                print("Episode %d took %d time steps" % (episode, t))
                times.append(int(t))
                break
        if len(times) < episode:
            times.append(TIME_STEPS)

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

    # plot distribution of episode times
    plt.plot(times)
    plt.title('Single Inverted Pendulum Episode Length Over Time')
    plt.xlabel('Episode Number')
    plt.ylabel('Timesteps')
    plt.show()

def select_action(env, state, explore_rate, q_table):
    # Select a random action
    if random.random() < explore_rate:
        action = int(env.action_space.sample()) + 3
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORATION, min(1.0, 1.0 - math.log10((t+1)/15)))

def get_learning_rate(t):
    return max(MIN_LEARNING, min(0.5, 1.0 - math.log10((t+1)/20)))

def state_to_bucket(state, STATE_BOUNDS, NUM_BUCKETS):
    bucket_indice = []
    #print(state)
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":
    main()
