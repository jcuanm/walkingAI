import gym
NUM_EPISODES = 1000
env = gym.make('Humanoid-v1')

for _ in range(NUM_EPISODES):
    observation = env.reset()

    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info =  env.step(action)

        # if our walker has fallen, end the episode
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
