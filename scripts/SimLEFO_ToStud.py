# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 20:06:34 2025

@author: haije001
"""
#%%
# Single perishable product
# Review period is 1 period
# Lead time is 1 period
# Order by base tsock policy with fixed level S = 2*mu + z * sqrt(2)*mu
#     if z = 1 / sqrt(2) then safety stock is demand for 1 day
# Discrete demand follows Poisson diistribution that is truncated at Dmx
# All consumer select the freshest items first (LEFO), unless disocunted items are availanble. 
#     If dicounted items (at rate aa%) are available then, aa% of customers will buy a disocunted item (FEFO)
# last-period discounting to all items: 
#     i.e., discount rate applies to all items that would expire if they remain unsold by the end of the period
#     a% discount makes a% of consumers buying disocunted instead of 'freshest' (FEFO instead of LEFO)
    
#%%
import numpy as np
from math import sqrt 
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pandas as pd
import random as rd

M = 5           # max. shelf life 
mu = 3          # mean demand of Poisson distribution
Dmx = 3*mu      # trunacte demand ceiling; you cannot realistically sell more
prob = pmf_values = poisson.pmf(np.arange(0,Dmx+1), mu)
prob[Dmx] = 1 - prob[0:Dmx].sum()

z = 1/sqrt(mu)   # for 1 day of safety stock set z = 1/sqrt(2)

# Dmx is the practical upperbound and 2 * mu + z * sqrt(2) * mu
# is a statistical motivated amount. Choose the lowest as the maximum
# number of items on the shelf per item age.
UBI = min( Dmx , 2 * mu + z * sqrt(2) * mu ) # Upperbound on \# items in each age class

r = 250         # regular sales price (excl. discount)
c = 175        # purchase price

Tw = 10         # warmin up period of simulation (is excluded in calculating statistics)
T = Tw + 36500  # length of a simulation (episode length).

#%% first define empty arrays that can be filled and re-initialized by zeros in a loop 
# (this prevents high RAM use until Garbage collection)
I = np.empty([T+1, M], dtype=int)
IL = np.empty([T, M], dtype=int)
BSPlvl = np.empty(T, dtype=int)
Order  = np.empty(T, dtype=int)
D  = np.empty(T, dtype=int)
DF  = np.empty(T, dtype=int)
DL  = np.empty(T, dtype=int)
DeltaF = np.empty([T, M], dtype=int)
DeltaL = np.empty([T, M], dtype=int)
SALES = np.empty(T, dtype=int)
PRF  = np.empty(T)
LS = np.empty(T, dtype=int)
WST  = np.empty(T, dtype=int)
Profit = np.zeros(11)

def State2Index(I):
    idx = I[0]
    for i in range(1,M):
        idx = (1+UBI)*idx + I[i]
    return idx

def Index2State(idx):
    Ilist = []
    for i in range(M):
        idx1 = int(idx/(1+UBI))
        Ilist.append(idx - idx1*(1+UBI))
        print(idx, idx1, Ilist)        
        idx = idx1
    return np.array(Ilist.reverse())
    
#%% EVALUATE FIXED LAST DAY DISCOUNTING BY LOOPING OVER DIFFERENT RATES 

for aa in  range(11):   #LOOP OVER DISCOUNT PERCENTAGES
  # RE-INITIALIZE variables and rand seeds
    rd.seed(42)
    np.random.seed(42)
    I[:] = np.zeros([T+1, M], dtype=int)
    IL[:] = np.zeros([T, M], dtype=int)
    BSPlvl[:]  = np.zeros(T, dtype=int)
    Order[:]  = np.zeros(T, dtype=int)
    D[:]  = np.zeros(T, dtype=int)
    DF[:]  = np.zeros(T, dtype=int)
    DL[:]  = np.zeros(T, dtype=int)
    DeltaF[:] = np.zeros([T, M], dtype=int)
    DeltaL[:] = np.zeros([T, M], dtype=int)
    SALES[:]  = np.zeros(T, dtype=int)
    PRF[:]  = np.zeros(T)
    LS[:]  = np.zeros(T, dtype=int)
    WST[:]  = np.zeros(T, dtype=int)
    
  # SIMULATE INVENTORY DYANMICS OVER ALL PERIODS
    for t in range(T):
        # observe state
        if t<0:
            print(I[t], State2Index(I[t]))
        # predict demand and set BSP level
        BSPlvl[t] = round(2*mu + z * sqrt(2)*mu)
        # set discount
        a = 0.05 * aa
        # set order quantity:
        Order[t] = max(0, BSPlvl[t] - I[t].sum())
        # set demand:
        D[t] = min( Dmx, np.random.poisson(mu) )
        # split demand 
        DF_frac = min(I[t,M-1], a * D[t] )
        DF_int = int(DF_frac)
        DF[t] = DF_int
        if DF_frac != DF_int: # stochastic rounding
            Frac = DF_frac - DF_int
            u = rd.random()
            if u < Frac: 
                DF[t] = DF_int + 1
        DL[t] = D[t] - DF[t]
        
        # pick FEFO demand:
        DeltaF[t,M-1] = DF[t]  # only oldest are discounted (<= max I[t,M-1]) 
        # available for LEFO customers/adjust stock levels:
        IL[t] = I[t] - DeltaF[t]    
        # pick LEFO demand:
        DeltaL[t,0] = min(DL[t], IL[t,0])   # youngest first
        for i in range(1,M):
            DeltaL[t,i] = min(DL[t] - DeltaL[t,0:i].sum() , IL[t,i]) 
            
        # adjust stock levels:
        I[t+1,:] = IL[t] - DeltaL[t]   
        WST[t] = I[t+1, M-1]
        I[t+1,:] = np.roll(I[t+1],1)    
        I[t+1,0] = Order[t]   # as lead time is one period
        # more stats:
        LS[t] = max( 0, D[t] - I[t].sum() )
        SALES[t] = DeltaF[t].sum() + DeltaF[t].sum()
        PRF[t] = r * DeltaL[t].sum() + r * (1-a) * DeltaF[t].sum() - c * Order[t]
     
  # COMPUTE & PRINT AVERAGES OF KEY PERFORMANCE MEASURES 
    WasteRel = WST[Tw:T].mean() / Order[Tw:T].mean()
    Fillrate = 1 - LS[Tw:T].mean() / D[Tw:T].mean()
    Profit[aa] = PRF[Tw:T].mean()
    print(f"Discount = {a*100:.0f}%, Profit = {Profit[aa]:.3f}, Waste = {WasteRel*100:.1f} %, Fillrate = {Fillrate*100:.1f} %")

#%%
plt.plot(5*np.arange(11), Profit, marker='o')  # Line plot with points
plt.title("Fixed last-day discounting")
plt.xlabel("discount %")
plt.ylabel("profit")
plt.grid(True)
plt.show()

# plt.plot(D)
# plt.show()
# plt.plot(LS)
# plt.show()

df = pd.DataFrame(
    {'I' : list(I[0:T]), # ,
     'BSPlvl': BSPlvl,
     'Order': Order,
     'D': D,
     'Delta': list(DeltaF + DeltaF),
     'LS': LS,
     'Wst' : WST,
     'Prf' : PRF
    } )

print(df.head(10))
print(df.tail(10))
