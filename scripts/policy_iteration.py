import numpy as np
from math import sqrt 

from enviroment import Environment

class PolicyIteration:
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.gamma = gamma
        # action space: [0.0, 0.05, ..., 0.50]
        self.actions = np.arange(env.discount_levels) * env.discount # we multiply discount levels times the discount
        self.nA = len(self.actions)
        # state space (1 + 8)^5 = 59049 states
        self.nS = int((1 + env.UBI) ** env.M)
        # base stock level (fixed every period)
        self.base_stock = round(2*env.mu + env.safety_factor * sqrt(2)*env.mu)
        # build compact transition table and R matrix
        self.build_model()
    
    def build_model(self):
        env = self.env
        # number of demand levels
        nD = env.max_demand + 1

        # compact transition table: for each (state, action, demand) store next state index
        self.next_ids_table = np.zeros((self.nS, self.nA, nD), dtype=int)
        # reward table: for each (state, action, demand) store reward
        self.reward_table = np.zeros((self.nS, self.nA, nD))
        # expected reward matrix (r[ids, ida])
        self.R = np.zeros((self.nS, self.nA))

        for ids in range(self.nS): # ids = 0: state = [0, 0, 0, 0, 0] ids = 59048: state = [8, 8, 8, 8, 8]
            state = env.index2state(ids)
            for ida, action in enumerate(self.actions):
                for d in range(nD):
                    next_state, reward = env.transition(state, action, d)
                    ids_next = int(env.state2index(next_state))
                    
                    # store in compact table
                    self.next_ids_table[ids, ida, d] = ids_next
                    self.reward_table[ids, ida, d] = reward
                    
                    # accumulate expected reward weighted by demand probability
                    self.R[ids, ida] += env.prob[d] * reward

    
    def run(self, theta=0.01, max_iter=1000):
        env = self.env
        nD = env.max_demand + 1
        
        # initialize
        V = np.zeros(self.nS)
        policy = np.zeros(self.nS, dtype=int)  # stores action index (ida)
        PolicyNotStable = True
        niterPI = 0
        
        while PolicyNotStable:
            niterPI += 1
            
            # IPE algorithm
            span = theta + 1
            niterIPE = 0
            while span > theta:
                niterIPE += 1
                V_prev = V.copy()
                for ids in range(self.nS):
                    ida = policy[ids]
                    V[ids] = 0
                    for d in range(nD):
                        ids_next = self.next_ids_table[ids, ida, d]
                        V[ids] += env.prob[d] * (self.reward_table[ids, ida, d]
                                                + self.gamma * V_prev[ids_next])
                Diff = V - V_prev
                span = max(Diff) - min(Diff)
                # in the IPE loop, after computing Diff
                span = max(Diff) - min(Diff)
                avg_reward = min(Diff)  # should converge to average reward per period
                print(f"min(Diff): {min(Diff):.4f}, max(Diff): {max(Diff):.4f}, span: {span:.4f}")
            
            print(f"PI iter {niterPI}: IPE converged in {niterIPE} sweeps, span={span:.4f}")
            
            # PI
            PolicyNotStable = False
            for ids in range(self.nS):
                qbest = -np.inf
                old_action = policy[ids]
                for ida in range(self.nA):
                    q = 0
                    for d in range(nD):
                        ids_next = self.next_ids_table[ids, ida, d]
                        q += env.prob[d] * (self.reward_table[ids, ida, d]
                                        + self.gamma * V_prev[ids_next])
                    if q > qbest:
                        qbest = q
                        policy[ids] = ida
                if policy[ids] != old_action:
                    PolicyNotStable = True
        
        print(f"Policy Iteration converged in {niterPI} iterations")
        self.V = V
        self.policy = policy
        return V, policy
