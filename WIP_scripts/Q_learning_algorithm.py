import numpy as np
import random
from math import ceil
import matplotlib.pyplot as plt

from scripts.environment import Environment


def q_learning_with_eval(epsilon, gamma, alpha, decay, epsilon_min, episodes, enviroment, eval_every=5000, eval_seed=999):
    # initialise empty Q-table
    q_table = np.zeros(shape=(ceil(enviroment.UBI + 1) ** enviroment.M, enviroment.discount_levels))
    # initialise empty list for eval_steps and rewards used for evaluation
    eval_steps, eval_rewards, eval_waste, eval_fill_rate = [], [], [], []
    # initialise steps at 0
    step = 0
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
                    epsilon=1, gamma=0.9, alpha=0.1, decay=0.9999, epsilon_min=0.05):
    q_table, eval_steps, eval_rewards, eval_waste, eval_fill_rate = q_learning_with_eval(
        epsilon=epsilon, gamma=gamma, alpha=alpha, decay=decay, epsilon_min=epsilon_min,
        episodes=episodes, enviroment=env, eval_every=eval_every, eval_seed=eval_seed,
    )
    profit = np.mean(eval_rewards[-5:])
    waste = np.mean(eval_waste[-5:])
    fill_rate = np.mean(eval_fill_rate[-5:])
    return profit, waste, fill_rate

# env = Environment(seed=1)
# q_table, eval_steps, eval_rewards, eval_waste, eval_fill_rate  = q_learning_with_eval(
#     epsilon=1, gamma=0.9, alpha=0.1, decay=0.9999, epsilon_min=0.05,
#     episodes=3, enviroment=env, eval_every=5000, eval_seed=2)
# print(f"profit: {round(np.mean(eval_rewards[-5:]), 3)}", f"waste: {round(np.mean(eval_waste[-5:]) * 100, 1)}%", f"fillrate: {round(np.mean(eval_fill_rate[-5:]) * 100, 1)}%" )

# plt.plot(eval_steps, eval_rewards, marker='o')
# plt.xlabel("Training step")
# plt.ylabel("Greedy eval reward (mean profit/step)")
# plt.title("Q-learning evaluation curve")
# plt.grid(True)
# plt.show()
