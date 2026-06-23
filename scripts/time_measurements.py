import time
import pandas as pd
from math import sqrt

from environment import Environment
from Q_learning_algorithm import run_q_learning
from sarsa_algorithm import run_sarsa
import dynamic_programming as dp


settings = [
    (3, 2, 1/sqrt(3)),    # 7^3   =       343 states
    (5, 3, 1/sqrt(3)),    # 9^5   =    59,049 states
    (5, 5, 0),            # 11^5  =   161,051 states
    (6, 3, 1/sqrt(3)),    # 9^6   =   531,441 states
    (5, 7, 0),            # 15^5  =   759,375 states
]


rl_kwargs = dict(episodes=10_000, convergence_tol=0.01, convergence_patience=3)

methods = {
    'Policy iteration': (dp.run_ipe_and_pi, {}),
    'Value iteration':  (dp.run_vi,          {}),
    'Q-learning':       (run_q_learning,     rl_kwargs),
    'Sarsa':            (run_sarsa,          rl_kwargs),
}

repetitions = 2
results = []

for M, mu, z in settings:
    env = Environment(max_shelf_life=M, mu=mu, z=z)
    
    for method_name, (run_method, kwargs) in methods.items():
        for rep in range(repetitions):
            if method_name in ('Policy iteration', 'Value iteration'):
                env.reset()
            start = time.perf_counter()
            result = run_method(env, **kwargs, **({'train_seed': rep} if kwargs else {}))
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
                'waste' : waste,
                'fill_rate' : fill_rate,
            })
            pd.DataFrame(results).to_csv('../data/timemeasurements_check.csv')
