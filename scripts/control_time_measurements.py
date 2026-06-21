from environment import Environment
import time
from math import sqrt
import pandas as pd

settings = [
    (3, 2, 1/sqrt(2)),    # 7^3   =       343 states
    (5, 3, 1/sqrt(3)),    # 9^5   =    59,049 states
    (5, 5, 0),            # 11^5  =   161,051 states
    (6, 3, 1/sqrt(3)),    # 9^6   =   531,441 states
    (5, 7, 0),            # 15^5  =   759,375 states
]

repetitions = 2
results = []

count = 0
for M, mu, z in settings:
    env = Environment(max_shelf_life=M, mu=mu, z=z)
    for rep in range(repetitions):
        profit = 0
        waste = 0
        fill_rate = 0
        elapsed_best = 0
        action_best = 0

        for action in range(env.discount_levels):
            start = time.perf_counter()


            env.reset()
            done = False
            while not done:
                _, _, done = env.step(action)
            if env.get_statistics()['profit'] > profit:

                profit = env.get_statistics()['profit']
                fill_rate = env.get_statistics()['fill_rate']
                waste = env.get_statistics()['waste']
                elapsed_best = time.perf_counter() - start
                action_best = action

        results.append({
            'M': M,
            'mu': mu,
            'z': z,
            'method': 'control',
            'rep': rep + 1,
            'time': elapsed_best,
            'action': action_best,
            'profit': profit,
            'waste' : waste,
            'fill_rate' : fill_rate,
        })

        print(results[count])
        pd.DataFrame(results).to_csv('../data/timemeasurements_control.csv')
        count += 1
