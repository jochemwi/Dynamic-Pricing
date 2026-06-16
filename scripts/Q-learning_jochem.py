import numpy as np
import random 
from math import ceil

from environment import Environment


def q_learning(epsilon, gamma, alpha, decay, epsilon_min, iterations, enviroment):
    
    # initialise the q table
    q_table = np.zeros(shape=(ceil(enviroment.UBI + 1) ** enviroment.M, enviroment.discount_levels))
    
    # loop
    for i in range(iterations):
        # get the state
        state = int(enviroment.reset()) # convert to integer
        done = False # set continue flag
        if i % 10 == 0: # only show every 10 itterations in terminal
            print(f"itteration: {i+1}/{iterations}")
        
        # only apply decay if above min_decoy
        if epsilon > epsilon_min:
            epsilon = epsilon * decay

        # continue while sim_length is not reached
        while done == False:
        
            # q value is value in the q_table at a specific state index
            q_value = q_table[int(state)]

            # greedy algorithm
            if epsilon > random.random():
                selection = random.randint(0, enviroment.discount_levels - 1)
            else:
                selection = q_value.argmax()
            
            # make a step
            next_state_idx, reward, done = enviroment.step(selection)

            # calculate q values (bellman eq.)
            current_q = q_table[state, selection]
            next_max_q = q_table[int(next_state_idx)].max()
            q_table[state, selection] = current_q + alpha * (reward + gamma * next_max_q - current_q)

            # update state
            state = int(next_state_idx)

    return q_table

def evaluate(q_table, environment):
    state = environment.reset()
    done = False

    while done == False:
        # q value is value in the q_table at a specific state index
        q_value = q_table[int(state)]
        selection = q_value.argmax()
        print(selection, selection * env.discount)
        
        # make a step
        next_state_idx, reward, done = environment.step(selection)
        
        state = int(next_state_idx)
    return environment.get_statistics()

env = Environment()

q_table = q_learning(epsilon=1, gamma=0.9, alpha=0.8, decay=0.9941, epsilon_min=0.05, iterations=50, enviroment=env)



state2test = [0, 0, 0, 0, 0]
print(q_table[int(env.state2index(state2test))].argmax())
print(evaluate(q_table, env))

profits = []
for action in range(env.discount_levels):
    env.reset()
    done = False
    while not done:
        _, _, done = env.step(action)
    profits.append(env.get_statistics()['profit'])

env.plot_env_results(profits)