import numpy as np
import math
import random
from matplotlib import pyplot as plt
import gym

from mujoco_inverted_pendulum import MujocoInvPend


class SarsaInvPend(MujocoInvPend):
    ## Learning related constants
    INIT_EXP = 0.8
    MIN_EXPLORATION = 1e-10
    EXP_COOLDOWN = 15

    INIT_LEARN = 0.6
    MIN_LEARNING = 1e-9
    LEARN_COOLDOWN = 20

    DISCOUNT = 0.99

    OFFSET = 3
    SCALE = 1
    NUM_ACTIONS = 2 * OFFSET + 1
    ACTION_CONSTRAINT = 0.1

    def __init__(self, env, explore=INIT_EXP, learn=INIT_LEARN):
        MujocoInvPend.__init__(self, env, explore, learn)

    def act(self, observation):
        """Follow epsilon greedy policy"""
        prev_state = self.state_to_bucket(observation)

        # Select a random action
        if random.random() < self.exploration_rate:
            action = int(self.env.action_space.sample()*self.SCALE) + self.OFFSET
        # Select the action with the highest q
        else:
            action = np.argmax(self.q_table[prev_state])
        return action

    def update(self, obs, reward):
        """Apply SARSA update function instead of Q learning and take action computed"""
        self.state = self.state_to_bucket(obs)
        next_action = self.act(obs)

        # Update the Q based on the result
        self.q_table[self.prev_state + (self.action,)] += self.learning_rate * (reward + self.DISCOUNT * self.q_table[self.state + (next_action,)] - self.q_table[self.prev_state + (self.action,)])

        self.action = next_action
        self.prev_state = self.state

    def reward(self, obs, prev_obs, default_reward):
        """Reward function"""
        reward = 1.0 - 3*(abs(obs[1] + obs[3])) - abs(obs[0] + obs[2])
        #print(reward)
        return reward


# OLD CODE

env = gym.make('CartPole-v1')

def get_reward(env_reward, observation, version):
    x = observation[0]
    x_hat = observation[1]
    angle = observation[2]
    angular_velocity = observation[3]
    
    if (version == 0):
        env_reward -= env_reward * (abs(angle) + abs(x))  
    elif (version == 1):
        env_reward -= env_reward * abs(angle)  
   
    return env_reward

def get_action(state, exploration, q_values):
    if random.random() >= exploration:
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
    time_steps = 1000
    times = []

    for i in range(num_episodes):
        observation = env.reset()
        initial_state = get_state(observation, bounds, num_states)
        action = get_action(initial_state, exploration, q_values)

        for t in range(time_steps):
            #env.render()
            observation, reward, done, info = env.step(action)
            reward = get_reward(reward, observation,0)
            state = get_state(observation, bounds, num_states)
            action_prime = get_action(state,exploration,q_values)
            
            # SARSA update
            q_values[initial_state + (action,)] += alpha*(reward + (gamma * q_values[state + (action_prime,)]) - q_values[initial_state + (action,)])
            initial_state = state
            action = action_prime
            #print_info(i,t,streaks,exploration,alpha)

            if done:
                times.append(int(t))
                # We consider an episode solved if it went 500 steps without falling
                if (t < 500):
                    streaks = 0
                else:
                    streaks += 1
                break

            elif t == 500:
                times.append(t)
                streaks += 1
                break

        exploration = max(.01, min(1, 1.0 - math.log10((i+1)/25)))
        alpha = max(.5, min(0.5, 1.0 - math.log10((t+1)/25))) # received this formula from a TF at office hours
    plt.title("SARSA Inverted pendulum time steps per episode")
    plt.xlabel('Episode')
    plt.ylabel('Time (t)')
    plt.plot(times, 'g')
    plt.show()

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
    train(500, bounds, num_states, q_values)

if __name__ == "__main__":
    main()
