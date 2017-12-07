import argparse
import gym
from matplotlib import pyplot as plt

def run_agent(env_id, agent_id, display):
    # instantiate environment
    env = gym.make(env_id)

    # select agent
    if agent_id == 'PDController':
        from pd_control_pendulum import PDController
        agent = PDController(env)
    elif agent_id == 'InvPend':
        from mujoco_inverted_pendulum import MujocoInvPend
        agent = MujocoInvPend(env)
    elif agent_id == 'SARSAInvPend':
        from sarsa_inv_pen import SarsaInvPend
        agent = SarsaInvPend(env)
    else:
        print("Not a valid agent")
        return

    episode_count = 300
    time_lim = 500

    times = []

    for i in range(episode_count):
        init = env.reset()
        ob = init
        agent.prev_state = agent.state_to_bucket(ob)
        t = 0
        while t < time_lim:
            if display:
                env.render()
            # act
            agent.act(ob)

            # step
            ob, reward, done, _ = agent.step(agent.action)

            reward = agent.reward(ob)

            # update
            agent.update(ob, reward)
            t += 1
            if done:
                print("Episode %d finished after %f time steps" % (i, t))
                times.append(int(t))
                break
            if len(times) < i:
                times.append(time_lim)
        agent.get_explore_rate(i)
        agent.get_learning_rate(i)

    # plot distribution of episode times
    plt.plot(times)
    plt.title('Single Inverted Pendulum Episode Length Over Time')
    plt.xlabel('Episode Number')
    plt.ylabel('Timesteps')
    plt.show()
    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='InvertedPendulum-v1', help='Select the environment to run')
    parser.add_argument('--agent_id', nargs='?', default='PDController', help='Select the agent to run')
    parser.add_argument('-display', action='store_true', help='Toggle rendering')
    args = parser.parse_args()

    run_agent(args.env_id, args.agent_id, args.display)


