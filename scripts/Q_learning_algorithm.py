import numpy as np
import random
from math import ceil
import matplotlib.pyplot as plt

from environment import Environment


def q_learning_with_eval(epsilon, gamma, alpha, decay, epsilon_min, episodes, enviroment,
                          eval_every=5000, eval_seed=999, convergence_tol=None, convergence_patience=3):
    # initialise empty Q-table
    q_table = np.zeros(shape=(ceil(enviroment.UBI + 1) ** enviroment.M + 1, enviroment.discount_levels))
    # initialise empty list for eval_steps and rewards used for evaluation
    eval_steps, eval_rewards, eval_waste, eval_fill_rate = [], [], [], []
    # initialise steps at 0
    step = 0
    converged = False
    # episode loop
    for i in range(episodes):
        state = int(enviroment.reset()) # set initials state
        done = False # is ittiration done flag
        while not done:
            # epsilon decay to stimulate exploration in early phase
            if epsilon > epsilon_min:
                epsilon = epsilon * decay
            # update q value for specific state
            q_value = q_table[int(state)]
            # epsilon logic
            if epsilon > random.random():
                action = random.randint(0, enviroment.discount_levels - 1)
            else:
                action = q_value.argmax()
            # get next_state_idx and reward after making selection
            next_state, reward, done = enviroment.step(action)

            # compute best q after determening current q
            current_q = q_table[int(state), int(action)]
            next_max_q = q_table[int(next_state)].max()
            # bellmans eq. for updating q table
            q_table[int(state), int(action)] = current_q + alpha * (reward + gamma * next_max_q - current_q)
            state = int(next_state)
            step += 1

            # evaluate greedy policy
            if step % eval_every == 0:
                eval_steps.append(step)
                reward_eval, waste_eval, fill_rate_eval = greedy_eval_reward(q_table, enviroment, eval_seed)
                eval_rewards.append(reward_eval)
                eval_waste.append(waste_eval)
                eval_fill_rate.append(fill_rate_eval)

                # early stopping: stop when last `convergence_patience` eval windows all change < tol
                if convergence_tol is not None and len(eval_rewards) >= convergence_patience + 1:
                    recent = eval_rewards[-(convergence_patience + 1):]
                    rel_diffs = [abs(recent[j] - recent[j-1]) / (abs(recent[j-1]) + 1e-8)
                                 for j in range(1, len(recent))]
                    if all(d < convergence_tol for d in rel_diffs):
                        converged = True
                        break
        if converged:
            break

    return q_table, eval_steps, eval_rewards, eval_waste, eval_fill_rate


def greedy_eval_reward(q_table, enviroment, eval_seed=999):

    
    np_state, rd_state = np.random.get_state(), random.getstate() # saves the random state to not mess with the training


    env = Environment(
        max_shelf_life=enviroment.M,
        mu=enviroment.mu,
        regular_sales_price=enviroment.regular_sales_price,
        purchase_price=enviroment.purchase_price,
        discount_levels=enviroment.discount_levels,
        discount=enviroment.discount,
        warm_up=enviroment.warm_up,
        z=enviroment.safety_factor,
        seed=eval_seed,
    )

    state = env.reset()
    done, total, n = False, 0.0, 0
    while not done:
        action = q_table[int(state)].argmax()   # greedy (no epsilon)
        state, reward, done = env.step(action)
        total += reward
        n += 1

    np.random.set_state(np_state)          # start at random position where we left
    random.setstate(rd_state)
    
    waste_pct = env.waste.sum() / env.order_quantity.sum()
    fill_rate = 1 - env.lost_sales.sum() / env.demand.sum()
    return total / n, waste_pct, fill_rate

def run_q_learning(env, episodes=3, eval_every=5000, eval_seed=2,
                    epsilon=1, gamma=0.9, alpha=0.1, decay=0.9999, epsilon_min=0.05,
                    convergence_tol=None, convergence_patience=3, train_seed=None):
    if train_seed is not None:
        random.seed(train_seed)
        np.random.seed(train_seed)
    q_table, eval_steps, eval_rewards, eval_waste, eval_fill_rate, total_steps = q_learning_with_eval(
        epsilon=epsilon, gamma=gamma, alpha=alpha, decay=decay, epsilon_min=epsilon_min,
        episodes=episodes, enviroment=env, eval_every=eval_every, eval_seed=eval_seed,
        convergence_tol=convergence_tol, convergence_patience=convergence_patience,
    )
    print('===== Q - learning =====')
    profit = np.mean(eval_rewards[-5:])
    waste = np.mean(eval_waste[-5:])
    fill_rate = np.mean(eval_fill_rate[-5:])
    return profit, waste, fill_rate, total_steps
