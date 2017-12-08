import gym
import numpy as np
import random
import math
from matplotlib import pyplot as plt
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

    def __init__(self, env):
        SarsaInvPend.__init__(self, env)


    def discretize(self):
        # Bounds for each discrete state
        state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        state_bounds = [(-1, 1) for _ in state_bounds]
        state_bounds[3] = 0.5, 2
        state_bounds[4] = 0.5, 2
        state_bounds[6] = -5, 5
        state_bounds[7] = -10, 10
        return state_bounds

    def reward(self, obs, prev_obs):
        #reward = - 0.1*abs(obs[1] - obs[2])**2 - 0.1*abs(obs[6] - obs[7])**2
        reward = abs((1-abs(obs[1]))/(obs[6]-prev_obs[6])) + abs((1-abs(obs[2]))/(obs[7]-prev_obs[7]))
        #print(reward)
        return reward


# OLD CODE

def init(name):
    env = gym.make(name)
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    return env, action_dim, observation_dim


NUM_EPISODES = 1000
TIME_STEPS = 1000

## Learning related constants
MIN_EXPLORE_RATE = .001
MIN_LEARNING_RATE = 0.01
DEBUG_MODE = False

def main():
    env, _, _ = init('InvertedDoublePendulum-v1')

    # Number of discrete states (bucket) per state dimension
    NUM_BUCKETS = (1, 10, 10, 10, 10, 1, 10, 10, 1, 1, 1) # x, sin, sin, cos, cos, v, w, w, , ,
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    STATE_BOUNDS = [(-1, 1) for _ in STATE_BOUNDS]
    STATE_BOUNDS[0] = -0.3, 0.3
    STATE_BOUNDS[3] = 0.3, 1
    STATE_BOUNDS[4] = 0.3, 1
    STATE_BOUNDS[6] = -5, 5
    STATE_BOUNDS[7] = -10, 10
    ## Creating a Q-Table for each state-action pair
    q_table = np.zeros(NUM_BUCKETS + (21,))


    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0
    times = []

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()
        state_0 = state_to_bucket(obv, STATE_BOUNDS, NUM_BUCKETS)
        action = select_action(env, state_0, explore_rate, q_table)

        for t in range(TIME_STEPS):
            #env.render()
            # Execute the action
            obv, reward, done, _ = env.step((action/3 + env.action_space.low)*0.5)
            
            # Observe the result
            state = state_to_bucket(obv, STATE_BOUNDS, NUM_BUCKETS)
            action_prime = select_action(env, state, explore_rate, q_table)

            # Update the Q based on the result
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor * q_table[state + (action_prime,)] - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state
            action = action_prime

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)

                print("")

            if done:
                print("Episode %d finished after %d time steps" % (episode, t))
                times.append(int(t))
                break


        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
    plt.plot(times, 'g')
    plt.title('Double Inverted Pendulum Episode Length Over Time')
    plt.xlabel('Episode Number')
    plt.ylabel('Timesteps')
    plt.show()

def select_action(env, state, explore_rate, q_table):
    # Select a random action
    if random.random() < explore_rate:
        action = int((env.action_space.sample() - env.action_space.low)*3)
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - 20*math.log10((t+1)/200)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - 0.5*math.log10((t+1)/400)))

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
