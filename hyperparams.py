from run_agent import run_agent
import numpy as np


def find_hyperparams(agent):
    """Grid Search to find optimal hyperparameter values
        -initial exploration rate
        -initial learning rate
    """
    times = run_agent(agent)
    return

if __name__ == '__main__':
    find_hyperparams('InvPend')