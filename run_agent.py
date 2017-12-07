import argparse
import gym

def run_agent(env_id, agent_id):
    env = gym.make(env_id)
    if agent_id == 'PDController':
        from pd_control_pendulum import PDController
        agent = PDController(env.action_space, env.observation_space)
    #elif agent_id == 'MujocoInvPend':
    #    from mujoco_inverted_pendulum import
    #    agent = (env.action_space, env.observation_space)
    else:
        print("Not a valid agent")
        return

    episode_count = 10
    reward = 0
    done = False

    for i in range(episode_count):
        init = env.reset()
        ob = init
        t = 0
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='InvertedPendulum-v1', help='Select the environment to run')
    parser.add_argument('agent_id', nargs='?', default='PDController', help='Select the agent to run')
    args = parser.parse_args()

    run_agent(args.env_id, args.agent_id)


