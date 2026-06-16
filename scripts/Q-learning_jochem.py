import numpy as np
import random 

from environment import Environment
from math import ceil

# q_table = np.zeros(shape=((int(env.UBI + 1)) ** env.M, env.discount_levels))
# state = env.reset()

# q_value = q_table[int(state)]

# epsilon = 0.2
# gamma = 0.9
# alpha = 0.5

# if epsilon > random.random():
#   selection = random.randit(0, env.discount_levels - 1)

# else:
#   selection = q_value.argmax()

# next_state_idx, reward, done = env.step(selection)

# if done != True:
#   current_q = q_table[state, selection]
#   next_max_q = q_table[next_state_idx].max()
#   q_table[state, selection] = current_q + alpha * (reward + gamma * next_max_q - current_q)

# state = next_state_idx

def q_learning(epsilon, gamma, alpha, decay, epsilon_min, iterations, enviroment):
    
    # make the tq table
    q_table = np.zeros(shape=(ceil(enviroment.UBI + 1) ** enviroment.M, enviroment.discount_levels))
    
    for i in range(iterations):
      # get the state
      state = int(enviroment.reset())
      done = False
      if i % 10 == 0: # every itteration seemed to much haha
        print(f"itteration: {i+1}/{iterations}")
      if epsilon > epsilon_min:
        epsilon = epsilon * decay

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

