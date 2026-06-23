#!/usr/bin/env python3
"""
Authors: Chris Ambagtsheer (student number: 1216414),
    Jochem Widdershoven (student number: )
Description: Customers prefer products as fresh as possible, so older
    products are eventually thrown out. To improve profit and reduce waste,
    the prices of the oldest products could be reduced by a discount.
    Depending on the product inventory composition, the optimal discount
    should be chosen to prevent full customer satisfaction with discounted
    products, while still reducing waste when selling full price products.
    In this repository, dynamic programming and temporal difference learning
    methods are applied, as well as a control where the discount is fixed.
    The data is generated for each run, where the next states are computed
    using the previous state and action. The customer behaviour is taken
    into account within the reward function.
Usage: python time_measurements.py [file_name]
    Where:
        python = python 3.0+ (program),
        time_measurements.py = this script,
        file_name = optional file name
"""


# import statements
import time
import pandas as pd
from math import sqrt
from environment import Environment
from Q_learning_algorithm import run_q_learning
from sarsa_algorithm import run_sarsa
import dynamic_programming as dp
from sys import argv

def get_variables():
    '''Create dataframes with settings to run on as well as the models.

    :return: the parameter settings, method descriptions and arguments for
        the reinforcement learning.
    '''
    settings = [
        (3, 2, 1 / sqrt(3)),  # 7^3   =       343 states
        (5, 3, 1 / sqrt(3)),  # 9^5   =    59,049 states
        (5, 5, 0),  # 11^5  =   161,051 states
        (6, 3, 1 / sqrt(3)),  # 9^6   =   531,441 states
        (5, 7, 0),  # 15^5  =   759,375 states
    ]

    rl_kwargs = {'episodes': 10_000, 'convergence_tol': 0.01,
                 'convergence_patience': 3}

    methods = {
        'Policy iteration': (dp.run_ipe_and_pi, {}),
        'Value iteration': (dp.run_vi, {}),
        'Q-learning': (run_q_learning, rl_kwargs),
        'Sarsa': (run_sarsa, rl_kwargs),
    }

    return settings, methods, rl_kwargs

def create_output(settings, methods, rlkwargs,
                  file = '../data/timemeasurements_check.csv'):
    '''Create a file with the model's output.

    :param settings: contains the model's settings, pd.dataframe.
    :param methods: contains the methods to evaluate, pd.dataframe.
    :param rlkwargs: contains the parameters for reinforcement learning,
        pd.dataframe.
    :param file: contains the file to save the output in, default =
        '../data/timemeasurements_check.csv'
    :return: None
    '''

    repetitions = 2
    results = []

    for M, mu, z in settings:
        env = Environment(max_shelf_life=M, mu=mu, z=z)

        for method_name, (run_method, kwargs) in methods.items():
            for rep in range(repetitions):
                if method_name in ('Policy iteration', 'Value iteration'):
                    env.reset()
                start = time.perf_counter()
                if kwargs:
                    result = run_method(env, **kwargs, train_seed=rep)
                else:
                    result = run_method(env)
                elapsed = time.perf_counter() - start
                profit, waste, fill_rate = result[:3]
                iterations = result[3] if len(result) > 3 else None
                results.append({
                    'M': M,
                    'mu': mu,
                    'z': z,
                    'method': method_name,
                    'rep': rep + 1,
                    'time': elapsed,
                    'iterations': iterations,
                    'profit': profit,
                    'waste': waste,
                    'fill_rate': fill_rate,
                })
                pd.DataFrame(results).to_csv(file)

def main():
    """Main function of the module."""

    if len(argv) < 1:
        raise ValueError('Incorrect use of script')
    elif len(argv) == 2:
        file = argv[1]
    else:
        file = '../data/timemeasurements_check.csv'

    settings, methods, rlkwargs = get_variables()
    create_output(settings, methods, rlkwargs, file)

    print('done')

if __name__ == '__main__':
    main()



