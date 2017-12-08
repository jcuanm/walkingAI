from run_agent import run_agent
import numpy as np


def find_hyperparams(agent):
    """Grid Search to find optimal hyperparameter values
        -initial exploration rate
        -initial learning rate
    """

    explore_list = list(np.linspace(0.1, 1.0, 10))
    learn_list = list(np.linspace(0.1, 1.0, 10))
    best = 0, 0, 0

    for explore in explore_list:
        for learn in learn_list:
            times = run_agent(agent, False, False, explore, learn)
            time = np.mean(times)
            if time > best[0]:
                best = time, explore, learn
    return best

if __name__ == '__main__':
    best = find_hyperparams('InvPend')
    print("Best time: %2f, best initial expl: %2f, best init learning: %2f" % best)
