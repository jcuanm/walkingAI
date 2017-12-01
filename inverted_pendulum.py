import numpy as np
import math
from random import random
import gym

env = gym.make('CartPole-v1')

def get_action(state, exploration, q_values):
    if random() >= exploration:
        action = np.argmax(q_values[state]) # Choose action w/ highest q
    else:
        action = env.action_space.sample() # Choose random action
    return action

def get_state(observation, bounds, num_states):
    indices = []
    for i in range(len(observation)):
        if observation[i] >= bounds[i][1]:
            index = num_states[i] - 1
        elif observation[i] <= bounds[i][0]:
            index = 0
        else:
            index = get_index(bounds, num_states, observation, i)
        indices.append(index)
    return tuple(indices)

def get_bounds(low, high):
    bounds = []
    for i in range(len(low)):
        bound = (low[i], high[i])
        bounds.append(bound)
    return bounds

def get_index(bounds, num_states, observation, i):
    deviation = bounds[i][1] - bounds[i][0]
    adjust = bounds[i][0] * (num_states[i] - 1) / deviation
    calibrate = (num_states[i] - 1) / deviation
    index = int(round(calibrate * observation[i] - adjust))
    return index

def print_info(ith_episode, t, streak, exploration, alpha):
    print("Episode = %d" % ith_episode)
    print("t = %d" % t)
    print("Exploration rate = %f" % exploration)
    print("Alpha = %f" % alpha)
    print("Streaks = %d" % streak)
    print("")

def train(num_episodes, bounds, num_states, q_values):
    streaks = 0
    alpha = .5
    exploration = 1
    gamma = 0.99  # the world is changing very little if at all
    time_steps = 250

    for i in range(num_episodes):
        observation = env.reset()
        initial_state = get_state(observation, bounds, num_states)

        for t in range(time_steps):
            env.render()
            action = get_action(initial_state, exploration, q_values)
            observation, reward, done, info = env.step(action)
            state = get_state(observation, bounds, num_states)
            max_Qvalue = np.amax(q_values[state])
            q_prime = initial_state + (action,)
            q_values[initial_state + (action,)] += alpha*(reward + gamma*(max_Qvalue) - q_values[q_prime])
            initial_state = state

            print_info(i,t,streaks,exploration,alpha)

            if done:
               # We consider an episode solved if it went 200 steps without falling
               if (t < 200):
                   streaks = 0
               else:
                   streaks += 1
               break

            elif t == 200:
                streaks += 1
                break

        exploration = max(.01, min(1, 1.0 - math.log10((i+1)/25)))
        alpha = max(.5, min(0.5, 1.0 - math.log10((t+1)/25))) # received this formula from a TF at office hours

        # Stop if we've succeesed 100 straight times 
        if streaks > 100:
            break

def main():
    num_position_states = 1
    num_velocity_states = 1
    num_angle_states = 6
    num_angular_velocity_states = 3
    num_states = (num_position_states, num_velocity_states, num_angle_states, num_angular_velocity_states)  # Number of discrete states per state dimension
    bounds = get_bounds(env.observation_space.low, env.observation_space.high) # Bounds per state
    bounds[1] = (-1, 1) # Setting the bounds for the cart
    bounds[3] = (-math.radians(50), math.radians(50)) # Setting the bounds for theta
    q_values = np.zeros((num_position_states, num_velocity_states, num_angle_states, num_angular_velocity_states,2)) # There are 2 discrete moves: left and right
    train(999, bounds, num_states, q_values)

if __name__ == "__main__":
    main()
