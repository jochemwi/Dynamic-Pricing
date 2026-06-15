import numpy as np
from scipy.stats import poisson

class SimpleInventoryEnv:
    def __init__(self, mu=3, c=175, g=250, z=1/np.sqrt(3)):
        self.mu = mu
        self.c = c
        self.g = g
        
        # base stock level B = 2μ + z√2 μ
        self.base_stock = int(2*mu + z*np.sqrt(2)*mu)
        
        # maximum inventory (UBI)
        self.max_inv = self.base_stock
        
        # state space S = {0,1,...,max_inv}
        self.states = np.arange(self.max_inv + 1)
        self.nS = len(self.states)
        
        # only 1 action: no discount
        self.actions = np.array([0.0])
        self.nA = len(self.actions)
        
        # demand distribution
        self.max_demand = mu * 3
        self.D = np.arange(self.max_demand + 1)
        self.p = poisson.pmf(self.D, mu)
        self.p[-1] = 1 - self.p[:-1].sum()  # ensure sum = 1

    def transition(self, s, a, d):
        """Scalar transition: next inventory after demand and replenishment."""
        order = max(0, self.base_stock - s)
        sales = min(s, d)
        next_s = max(0, min(self.max_inv, s + order - d))
        reward = self.g * sales - self.c * order
        return next_s, reward

def build_dp_matrices(env):
    P = np.zeros((env.nA, env.nS, env.nS))
    R = np.zeros((env.nS, env.nA))
    
    for ids, s in enumerate(env.states):
        for ida, a in enumerate(env.actions):
            for d, pd in zip(env.D, env.p):
                s1, r = env.transition(s, a, d)
                ids1 = int(s1)
                
                P[ida, ids, ids1] += pd
                R[ids, ida] += pd * r
    
    return P, R