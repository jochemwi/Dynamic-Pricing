import time
import pandas as pd

from environment import Environment
from Q_learning_algorithm import run_q_learning
from sarsa_algorithm import run_sarsa

# settings = [
#     (5, 3, 3),
#     (5, 6, 0),
#     (5, 9, 0),
#     (6, 3, 3),
#     (7, 3, 3),
# ]

settings = [(1, 1, 1)]

methods = {
    # 'Policy iteration': run_policy_iteration,
    # 'Value iteration':  run_value_iteration,   
    'Q-learning':       run_q_learning,         
    'Sarsa':            run_sarsa,
}

repetitions = 2
results = []

for M, mu, z in settings:
    env = Environment(max_shelf_life=M, mu=mu, z=z)

    for method_name, run_method in methods.items():
        for rep in range(repetitions):
            start = time.perf_counter()
            policy, profit, waste, fill_rate = run_method(env)
            elapsed = time.perf_counter() - start
            results.append({
                'M': M,
                'mu': mu,
                'z': z,
                'method': method_name,
                'rep': rep + 1,
                'time': elapsed,
                'profit': profit,
                'waste' : waste,
                'fill_rate' : fill_rate,
            })

results_df = pd.DataFrame(results)
results_df.to_csv('./data/timemeasurements.csv')
