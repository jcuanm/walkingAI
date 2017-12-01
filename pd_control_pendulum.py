import argparse
import gym


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, initial, observation, reward, done):
        p_kp = 4
        p_kd = 1
        c_kp = 0.1
        c_kd = 0.2
        x = observation[0] - initial[0]
        theta = observation[1] - initial[1]
        v = observation[2] - initial[2]
        omega = observation[3] - initial[3]
        action = p_kp * theta + p_kd * omega + c_kp * x + c_kd * v
        if action > self.action_space.high:
            action = self.action_space.high
        elif action < self.action_space.low:
            action = self.action_space.low
        return action

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='InvertedPendulum-v1', help='Select the environment to run')
    args = parser.parse_args()

    env = gym.make(args.env_id)
    agent = RandomAgent(env.action_space)

    episode_count = 10
    reward = 0
    done = False

    for i in range(episode_count):
        init = env.reset()
        ob = init
        t=0
        while True:
            env.render()
            action = agent.act(init, ob, reward, done)
            ob, reward, done, _ = env.step(action)
            t += 1
            if done:
                print("Episode %d finished after %f time steps" % (i, t))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
