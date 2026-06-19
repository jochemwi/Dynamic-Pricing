import numpy as np
from math import sqrt

from scripts.environment import Environment

class PolicyIteration:
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.gamma = gamma
        # action space: [0.0, 0.05, ..., 0.50]
        self.actions = np.arange(env.discount_levels) * env.discount
        self.nA = len(self.actions)
        # number of states: (UBI+1)^M
        self.nS = int((1 + env.UBI) ** env.M)
        # build compact transition table and R matrix
        self.build_model()
    
    def build_model(self):
        env = self.env
        nD = env.max_demand + 1
        
        print("Building R and transition tables...")
        
        # expected reward matrix
        self.R = np.zeros((self.nS, self.nA))
        # compact transition table
        self.next_ids_table = np.zeros((self.nS, self.nA, nD), dtype=int)
        
        for ids in range(self.nS):
            state = env.index2state(ids)
            for ida, action in enumerate(self.actions):
                for d in range(nD):
                    next_state, reward = env.transition(state, action, d)
                    ids_next = int(env.state2index(next_state))
                    self.next_ids_table[ids, ida, d] = ids_next
                    self.R[ids, ida] += env.prob[d] * reward
        
        print("Done.")
    
    def run(self, theta=0.01, max_iter=1000):
        env = self.env
        nD = env.max_demand + 1
        
        # initialize like notebook
        V = np.zeros(self.nS)
        policy = np.zeros(self.nS, dtype=int)
        PolicyNotStable = True
        niterPI = 0
        
        while PolicyNotStable:
            niterPI += 1
            
            # --- Iterative Policy Evaluation (IPE) ---
            span = theta + 1
            niterIPE = 0
            while span > theta:
                niterIPE += 1
                V_prev = V.copy()
                for ids in range(self.nS):
                    ida = policy[ids]
                    # expected reward already precomputed in R[ids, ida]
                    # add discounted expected future value over all demand levels
                    V[ids] = self.R[ids, ida]
                    for d in range(nD):
                        ids_next = self.next_ids_table[ids, ida, d]
                        V[ids] += env.prob[d] * self.gamma * V_prev[ids_next]
                Diff = V - V_prev
                span = max(Diff) - min(Diff)
            
            avg_reward = min(Diff)
            print(f"PI iter {niterPI}: IPE converged in {niterIPE} sweeps, span={span:.4f}, avg_reward={avg_reward:.4f}")
            
            # --- Policy Improvement (like notebook) ---
            PolicyNotStable = False
            for ids in range(self.nS):
                old_action = policy[ids]
                qbest = -np.inf
                for ida in range(self.nA):
                    # q value = expected reward + discounted expected future value
                    q = self.R[ids, ida]
                    for d in range(nD):
                        ids_next = self.next_ids_table[ids, ida, d]
                        q += env.prob[d] * self.gamma * V_prev[ids_next]
                    if q > qbest:
                        qbest = q
                        policy[ids] = ida
                if policy[ids] != old_action:
                    PolicyNotStable = True
        
        print(f"Policy Iteration converged in {niterPI} iterations")
        self.V = V
        self.policy = policy
        return V, policy