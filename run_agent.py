import argparse
import gym
from matplotlib import pyplot as plt

def run_agent(agent_id, display):
    # select agent
    if agent_id == 'PDController':
        from pd_control_pendulum import PDController
        env = gym.make('InvertedPendulum-v1')
        agent = PDController(env)
        episode_count = 300
    elif agent_id == 'InvPend':
        from mujoco_inverted_pendulum import MujocoInvPend
        env = gym.make('InvertedPendulum-v1')
        agent = MujocoInvPend(env)
        episode_count = 300
    elif agent_id == 'SarsaInvPend':
        from sarsa_inv_pen import SarsaInvPend
        env = gym.make('InvertedPendulum-v1')
        agent = SarsaInvPend(env)
        episode_count = 300
    elif agent_id == 'DoubleInvPend':
        from double_inv_pend import DoubleInvPend
        env = gym.make('InvertedDoublePendulum-v1')
        agent = DoubleInvPend(env)
        episode_count = 10000
    elif agent_id == 'SarsaDoubleInvPend':
        from sarsa_double_inv import SarsaDoubleInvPend
        env = gym.make('InvertedDoublePendulum-v1')
        agent = SarsaDoubleInvPend(env)
        episode_count = 10000
    else:
        print("Not a valid agent")
        return

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

            prev_ob = ob

            # step
            ob, reward, done, _ = agent.step(agent.action)

            reward = agent.reward(ob, prev_ob)

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
    plt.title('Episode Length Over Time')
    plt.xlabel('Episode Number')
    plt.ylabel('Timesteps')
    plt.show()
    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--agent_id', nargs='?', default='PDController', help='Select the agent to run')
    parser.add_argument('-display', action='store_true', help='Toggle rendering')
    args = parser.parse_args()

    run_agent(args.agent_id, args.display)


