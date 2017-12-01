import gym
import numpy as np
import random
import math
from matplotlib import pyplot as plt

def init(name):
    env = gym.make(name)
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    return env, action_dim, observation_dim

NUM_EPISODES = 500
TIME_STEPS = 1000

## Learning related constants
MIN_EXPLORATION = 1e-8
MIN_LEARNING = 0.001
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
            if episode > 498:
                env.render()

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

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

    # plot distribution of episode times
    plt.plot(times)
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
    return max(MIN_EXPLORATION, min(1.0, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING, min(0.3, 1.0 - math.log10((t+1)/25)))

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