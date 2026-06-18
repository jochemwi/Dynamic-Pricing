import numpy as np
import random 
from math import ceil
import matplotlib.pyplot as plt

from environment import Environment


def sarsa(epsilon, gamma, alpha, decay, epsilon_min, episodes, enviroment):
    
    # initialise the q table
    q_table = np.zeros(shape=(ceil(enviroment.UBI + 1) ** enviroment.M, enviroment.discount_levels))
    rewards = []
    # loop
    for i in range(episodes):
        # get the state
        state = int(enviroment.reset()) # convert to integer
        done = False # set continue flag
        if i % 10 == 0: # only show every 10 itterations in terminal
            print(f"episode: {i+1}/{episodes}")

        # continue while sim_length is not reached
        while done == False:

            # only apply decay if above min_decay (per step, episodes are very long)
            if epsilon > epsilon_min:
                epsilon = epsilon * decay

            # q value is value in the q_table at a specific state index
            q_value = q_table[int(state)]

            # greedy algorithm
            if epsilon > random.random():
                selection = random.randint(0, enviroment.discount_levels - 1)
            else:
                selection = q_value.argmax()
            
            # make a step
            next_state_idx, reward, done = enviroment.step(selection)

            next_q_value = q_table[int(next_state_idx)]
            if epsilon > random.random():
                action = random.randint(0, enviroment.discount_levels - 1)
            else:
                action = next_q_value.argmax()
            
            next_max_q = q_table[int(next_state_idx), action]

            # calculate q values (bellman eq.)
            current_q = q_table[state, selection]
            q_table[state, selection] = current_q + alpha * (reward + gamma * next_max_q - current_q)

            # update state
            state = int(next_state_idx)

            # store reward per iteration (step), not per episode
            rewards.append(reward)
    return q_table, rewards

def evaluate(q_table, environment):
    state = environment.reset()
    done = False

    while done == False:
        # q value is value in the q_table at a specific state index
        q_value = q_table[int(state)]
        selection = q_value.argmax()
        
        # make a step
        next_state_idx, reward, done = environment.step(selection)
        
        state = int(next_state_idx)
    return environment.get_statistics()