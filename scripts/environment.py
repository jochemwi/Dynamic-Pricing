import numpy as np
from math import sqrt 
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pandas as pd
import random as rd

class Environment():
    def __init__(
            self, 
            max_shelf_life=5,           # number of days before expiry
            mu=3,                       # mean daily demand
            regular_sales_price=250,    # selling price without discount
            purchase_price=175,         # cost per item ordered
            discount_levels=11,         # number of discount levels (0% to 50%)
            discount = 0.05,
            warm_up = 10,               # period after which statistics start 
            z = 1/sqrt(3)               # safety factor for base stock level
            ):
        
        self.M = max_shelf_life
        self.mu = mu
        self.max_demand = mu * 3                        # maximize demand distribution at 3x mean
        self.regular_sales_price = regular_sales_price
        self.purchase_price = purchase_price
        self.discount_levels = discount_levels
        self.discount = discount          
        self.safety_factor = z
        self.warm_up = warm_up
        self.sim_length = self.warm_up + 36500          # total simulation length including
        self.prob = self.probability()                  # demand probabilities follwing poisson
        self.UBI = min(self.max_demand, 2*self.mu + self.safety_factor * sqrt(2)*self.mu)
        self.base_stock = round(2*self.mu + self.safety_factor * sqrt(2)*self.mu)
        self.reset()                                    # initialize arrays on initilisation Enviroment
    
    def probability(self):
        # compute truncated Poisson demand probabilities
        prob = poisson.pmf(np.arange(0, self.max_demand + 1), self.mu)
        # last value absorbs all remaining probability mass to ensure sum = 1
        prob[self.max_demand] = 1 - prob[0:self.max_demand].sum()
        return prob

    def reset(self):
        # random seed for repproducability
        rd.seed(42)
        np.random.seed(42)
        # initialize all arrays to zero
        self.inventory_matrix = np.zeros([self.sim_length+1, self.M], dtype=int)            # inventory matrix
        self.inventory_after_fefo = np.zeros([self.sim_length, self.M], dtype=int)          # inventory after FEFO demand is met
        self.base_stock_level = np.zeros(self.sim_length, dtype=int)                        # base stock level per period
        self.order_quantity = np.zeros(self.sim_length, dtype=int)                          # order quantity per period
        self.demand = np.zeros(self.sim_length, dtype=int)                                  # total demand per period
        self.fefo_demand = np.zeros(self.sim_length, dtype=int)                             # FEFO demand
        self.lefo_demand = np.zeros(self.sim_length, dtype=int)                             # LEFO demand
        self.items_picked_fefo = np.zeros([self.sim_length, self.M], dtype=int)             # actual items picked by FEFO customers per age class
        self.items_picked_lefo = np.zeros([self.sim_length, self.M], dtype=int)             # actual items picked by LEFO customers per age class
        self.sales = np.zeros(self.sim_length, dtype=int)                                   # total sales per period
        self.profit  = np.zeros(self.sim_length)                                            # profit per period
        self.lost_sales = np.zeros(self.sim_length, dtype=int)                              # lost sales per period
        self.waste = np.zeros(self.sim_length, dtype=int)                                   # waste per period
        self.t = 0                                                                          # current time step
        return self.state2index(self.inventory_matrix[0])                                                # return initial state (all zeros) ([0,0,0,0,0] with m = 5)
    
    def split_demand(self, action): # action is fraction of demand atracted to discount
         # compute how many FEFO customers buy discounted oldest items
         # action = discount rate, so action * D[t] = fraction of demand attracted to discounted items
         # cannot exceed available oldest stock I[t, M-1]
         # sefo

        self.fefo_demand_frac = min(self.inventory_matrix[self.t, self.M-1], action * self.demand[self.t])
        self.fefo_demand_int = int(self.fefo_demand_frac)
        self.fefo_demand[self.t] = self.fefo_demand_int
        # stochastic rounding: round up with probability equal to fractional part
        if self.fefo_demand_frac != self.fefo_demand_int:
            frac = self.fefo_demand_frac - self.fefo_demand_int
            if rd.random() < frac:
                self.fefo_demand[self.t] = self.fefo_demand_int + 1
        # remaining demand goes to LEFO customers (buy fresh)
        self.lefo_demand[self.t] = self.demand[self.t] - self.fefo_demand[self.t]


    def update_inventory(self):
        # FEFO customers only pick from oldest age class (M-1)
        self.items_picked_fefo[self.t, self.M-1] = self.fefo_demand[self.t]
        # available for LEFO customers
        self.inventory_after_fefo[self.t] = self.inventory_matrix[self.t] - self.items_picked_fefo[self.t] # example [4,2,2,2,6] - [0,0,0,0,1] = [4,2,2,2,5]
        # LEFO customers buy freshest first (age 0), capped by available stock
        self.items_picked_lefo[self.t, 0] = min(self.lefo_demand[self.t], self.inventory_after_fefo[self.t, 0])
        # if slot 0 didn't cover full LEFO demand, spill into progressively older slots
        for i in range(1, self.M):
            self.items_picked_lefo[self.t, i] = min(self.lefo_demand[self.t] - self.items_picked_lefo[self.t, 0:i].sum(), self.inventory_after_fefo[self.t, i])
        # next inventory = remaining stock after all picks (LEFO and FEFO)
        self.inventory_matrix[self.t+1, :] = self.inventory_after_fefo[self.t] - self.items_picked_lefo[self.t] # results in a matrix 
        # waste = oldest items still unsold (they expire at end of period)
        self.waste[self.t] = self.inventory_matrix[self.t+1, self.M-1] # What is still in slot M-1 at this point expires tonight
        # age all items by one day (shift columns right)
        self.inventory_matrix[self.t+1, :] = np.roll(self.inventory_matrix[self.t+1], 1) # last items move to the first, but will be replaced in the next step
        # new order arrives in slot 0 (lead time = 1 period)
        self.inventory_matrix[self.t+1, 0] = self.order_quantity[self.t]

    def compute_stats(self, action, p = None):
        # lost sales = demand that could not be met from available stock
        self.lost_sales[self.t] = max(0, self.demand[self.t] - self.inventory_matrix[self.t].sum())
        # total sales = all items picked by both customer types (FEFO + LEFO)
        self.sales[self.t] = self.items_picked_fefo[self.t].sum() + self.items_picked_lefo[self.t].sum()
        # profit = regular sales revenue + discounted sales revenue - purchase cost
        self.profit[self.t] = (self.regular_sales_price * self.items_picked_lefo[self.t].sum()
                          + self.regular_sales_price * (1 - action) * self.items_picked_fefo[self.t].sum()
                          - self.purchase_price * self.order_quantity[self.t])
        if p:
            self.profit[self.t] = self.profit[self.t] * p
        
    def get_statistics(self):
        # compute performance metrics excluding warm-up period
        waste_rel = self.waste[self.warm_up:].mean() / self.order_quantity[self.warm_up:].mean()
        fill_rate = 1 - self.lost_sales[self.warm_up:].mean() / self.demand[self.warm_up:].mean()
        profit = self.profit[self.warm_up:].mean()
        return {
            'profit': profit,
            'waste': waste_rel,
            'fill_rate': fill_rate
        }

    def step(self, action):
        # convert discrete action integer to discount fraction
        discount_fraction = action * self.discount  # e.g. action=3 → 0.15 → 15% discount

        self.base_stock_level[self.t] = round(2*self.mu + self.safety_factor * sqrt(2)*self.mu)
        self.order_quantity[self.t] = max(0, self.base_stock_level[self.t] - self.inventory_matrix[self.t].sum())
        self.demand[self.t] = min(self.max_demand, np.random.poisson(self.mu))
        
        self.split_demand(discount_fraction)
        self.update_inventory()
        self.compute_stats(discount_fraction)

        reward = self.profit[self.t]
        next_state_vec = self.inventory_matrix[self.t + 1]
        next_state_idx = self.state2index(next_state_vec)  # return scalar index

        self.t += 1
        done = self.t >= self.sim_length
        return next_state_idx, reward, done

    def get_probability_matrix(self, SS, PP, inventory_matrix, ida, ids, action):
        '''Create a probability matrix for next states (inventory)

        :param SS: np.array of shape (nS, M). Contains all possible inventories.
        :param PP: np.array of shape (nA, nS, nS). Contains the probabilities
            of the next state given the current state and current action.
        :param inventory_matrix: np.array of shape (M). Contains the current
            inventory.
        :param ida: int. The index of current action in the action matrix.
        :param action: int. Current action.
        :return: Updated probability matrix,
            type: np.array with shape (nA, nS, nS).
        '''

        self.inventory_matrix[self.t,:] = inventory_matrix

        discount_fraction = action * self.discount
        self.base_stock_level[self.t] = round(2 * self.mu + self.safety_factor * sqrt(2) * self.mu)
        self.order_quantity[self.t] = max(0, self.base_stock_level[self.t] - self.inventory_matrix[self.t].sum())

        PP, tot_profit = self.demand_profit(SS, PP, ida, ids, discount_fraction)
        return PP, tot_profit

    def demand_profit(self, SS, PP, ida, ids, discount_fraction):
        tot_profit = 0

        for idd, d in enumerate(range(self.max_demand + 1)):  # demand
            self.demand[self.t] = d

            self.split_demand(discount_fraction)
            self.update_inventory()

            next_state_vec = self.inventory_matrix[self.t + 1]
            ids1 = np.where((SS == next_state_vec).all(axis=1))[0][0].item()

            p = self.prob[idd]
            PP[ida, ids, ids1] += p
            self.compute_stats(discount_fraction, p)
            tot_profit += self.profit[self.t]

        return PP, tot_profit

    def state2index(self, state):
        idx = state[0]
        for i in range(1, self.M):
            idx = (1 + self.UBI) * idx + state[i]
        return idx

    def index2state(self, idx):
        Ilist = []
        for i in range(self.M):
            idx1 = int(idx / (1 + self.UBI))
            Ilist.append(idx - idx1 * (1 + self.UBI))
            idx = idx1
        Ilist.reverse()
        return np.array(Ilist)

    def plot_env_results(self, profit):
        # plot fixed last-day discounting
        step = self.discount * 100
        # step = 100 / (self.discount_levels - 1) # automatically changes with discount levels instead of calculating range and fraction
        plt.plot(step * np.arange(self.discount_levels), profit,  marker='o')
        plt.title("Fixed last-day discounting")
        plt.xlabel("discount %")
        plt.ylabel("profit")
        plt.grid(True)
        plt.show()

    def __repr__(self):
        # create readable summary
        return (f"Environment("
                f"M={self.M}, "
                f"mu={self.mu}, "
                f"Dmx={self.max_demand}, "
                f"regular_sales_price={self.regular_sales_price}, "
                f"purchase_price={self.purchase_price}, "
                f"discount_levels={self.discount_levels}, "
                f"safety_factor={self.safety_factor:.3f}, "
                f"warm_up={self.warm_up}, "
                f"sim_length={self.sim_length})")


# # plot
# env = Environment()
# profits = []
# for action in range(env.discount_levels):
#     env.reset()
#     done = False
#     while not done:
#         _, _, done = env.step(action)
#     profits.append(env.get_statistics()['profit'])
#
# env.plot_env_results(profits)