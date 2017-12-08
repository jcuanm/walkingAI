import argparse
import gym
import numpy as np
from matplotlib import pyplot as plt


def run_agent(agent_id, display=False, plot=True, init_expl=False, init_learn=False):
    # select agent
    if agent_id == 'PDController':
        from pd_control_pendulum import PDController
        env = gym.make('InvertedPendulum-v1')
        agent = PDController(env)
        episode_count = 10
        time_lim = 950
    elif agent_id == 'InvPend':
        from mujoco_inv_pend import MujocoInvPend
        env = gym.make('InvertedPendulum-v1')
        agent = MujocoInvPend(env)
        episode_count = 300
        time_lim = 950
    elif agent_id == 'SarsaInvPend':
        from sarsa_inv_pen import SarsaInvPend
        env = gym.make('InvertedPendulum-v1')
        agent = SarsaInvPend(env)
        episode_count = 300
        time_lim = 950
    elif agent_id == 'DoubleInvPend':
        from double_inv_pend import DoubleInvPend
        env = gym.make('InvertedDoublePendulum-v1')
        agent = DoubleInvPend(env)
        episode_count = 20000
        time_lim = 500
    elif agent_id == 'SarsaDoubleInvPend':
        from sarsa_double_inv import SarsaDoubleInvPend
        env = gym.make('InvertedDoublePendulum-v1')
        agent = SarsaDoubleInvPend(env)
        episode_count = 20000
        time_lim = 500
    elif agent_id == 'Walker':
        from walker import Walker
        env = gym.make('Walker2d-v1')
        agent = Walker(env)
        episode_count = 1000
        time_lim = 500
    else:
        print("Not a valid agent")
        return

    if init_expl and init_learn:
        agent.exploration_rate = init_expl
        agent.learning_rate = init_learn

    # keep track of length of episodes for plotting
    times = []
    obs = np.array([])

    for i in range(episode_count):
        init = env.reset()
        ob = init
        agent.prev_state = agent.state_to_bucket(ob)
        t = 0
        while t < time_lim:
            # only display when told to
            if display:
                env.render()
            # keep track of observations of final episode
            if i == episode_count - 1:
                obs = np.append(obs, ob[1])

            # act
            agent.act(ob)

            # hold on to previous observation if needed
            prev_ob = ob

            # step
            ob, reward, done, _ = agent.step(agent.action)

            # reward function approximation
            reward = agent.reward(ob, prev_ob, reward)

            # update
            agent.update(ob, reward)
            t += 1
            if done:
                print("Episode %d: %f time steps" % (i, t))
                times.append(int(t))
                break
            if len(times) < i:
                times.append(time_lim)

        # cool-down for exploration and learning
        agent.get_explore_rate(i)
        agent.get_learning_rate(i)

    if plot:
        # plot distribution of episode times
        plt.figure()
        plt.plot(times)
        plt.title('Episode Length Over Time')
        plt.xlabel('Episode Number')
        plt.ylabel('Timesteps')

        plt.figure()
        print(obs)
        plt.plot(obs)
        plt.title('Pendulum Angle Over Time')
        plt.xlabel('Timesteps')
        plt.ylabel('Angle')
        plt.show()
    # Close the env and write monitor result info to disk
    env.close()
    return times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--agent_id', nargs='?', default='PDController', help='Select the agent to run')
    parser.add_argument('-display', action='store_true', help='Toggle rendering')
    parser.add_argument('-no_plot', action='store_false', help='Toggle plot')
    args = parser.parse_args()

    run_agent(args.agent_id, args.display, args.no_plot)


