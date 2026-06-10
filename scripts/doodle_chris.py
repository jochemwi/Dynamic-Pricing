import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pandas as pd
import random as rd

M = 5 # max shelf life
mu = 3 # mean demand of Poisson distribution
Dmx = 3 * mu # truncate demand distribution: deman in {0,1,2,...,dmx}

# np.arange creates an array of length 0 : Dmx + 1
# poisson.pmf(k, mu) indicates size of event and mean (also variance)
# hence, the probability for each value of 0 to 9 is given when the mean
# value is 3.
prob = pmf_values = poisson.pmf(np.arange(0, Dmx + 1), mu)

# you replace the probability at index 9 (which is very low) with
# the summed probabilities of index 0 to 8. That way, the total
# probability of finding any value between 0 and 9 is captured
# in these data.
prob[Dmx] = 1 - prob[0:Dmx].sum()

# safety stock is the amount of product you keep to prevent you from
# running out of stock due to demand uncertainty.
# safety stock = z * sigma / L^-(1/2).
# sigma is standard deviation of demand and L is lead time.

# by setting z = 1/2^(1/2), a single day of safety stock is kept in
# inventory.
# See: sigma = mu^(1/2) because of Poisson model
# See: L = 1 because one day of safety stock
# See: safety stock should be mu/2 for one day.
# (variance is the squared standard deviation, and variance = mean
# under Poisson distribution).
z = 1 / sqrt(2) # for 1 day of safety stock, set z = 1 / sqrt(2)

# return the lowest value; either the truncated demand distribution
# or the other value. Set this value as the upper bound number of items.
# the upper bound of items is;
# mean + z * standard deviation
# worse case, two days are put into one day due to uncertain
# restock times. So, this is statistically the max items to keep.
# see: sigma = sqrt(mu), z = 1/2^(1/2), mu = 3, days = 2 so multiply
# mu and sigma with 2 (hence sqrt(2)).
UBI = min(Dmx, 2 * mu + z * 2 * sqrt(mu))

r = 250 # regular sales price (excl. discount)
c = 175 # purchase price. So baseline margin is 250 - 175 = 75 per unit.

Tw = 10 # warming up period of simulation (is excluded in calculating statistics)

# One episode contains 36500 days; hence 100 years. Each step is one day.
T = Tw + 36500 # length of a simulation (episode length).

#####################################################################
# define empty arrays to fill and reinitialize by zeros in a loop.  #
# that way, Python does not have to fragment RAM by appending items #
# to a list but can simply overwrite existing and reserved memory.  #
#####################################################################

##INVENTORY STATE
# Inventory at start of each period, tracked per age class [1 to M].
# it has T + 1 rows as you need opening and closing states.
I = np.empty([T + 1, M], dtype = int)

# Inventory Lost (leftover); items remaining per age class after demandis
# realised and before disposal.
IL = np.empty([T, M], dtype = int)

# Delta FEFO; change in inventory for the oldest products at day's end.
DeltaF = np.empty([T, M], dtype=int)

# Delta LEFO; change in inventory for the fresher products at day's end
DeltaL = np.empty([T, M], dtype=int)

## DEMAND TRACKING
# Demand; total demand arriving each period
D  = np.empty(T, dtype=int)

# Demand FEFO; demanded satisfied from older items
# likely at a discount.
DF  = np.empty(T, dtype=int)

# Demanded LEFO; demanded satisfied from newer items.
DL  = np.empty(T, dtype=int)

## INVENTORY DECISIONS
# Base Stock Policy Level: the target inventory level your ordering
# policy aims to reach each period.
BSPlvl = np.empty(T, dtype = int)

# units ordered each period.
Order = np.empty(T, dtype = int)

## OUTCOMES PER PERIOD
# Units actually sold
SALES = np.empty(T, dtype = int)

# Profit per period.
PRF = np.empty(T)

# Lost sales; demand that arrived but couldn't be fulfilled (stock out)
LS = np.empty(T, dtype = int)

# Waste; units that expired unsold and had to be discarded.
WST = np.empty(T, dtype = int)

# Average profit across 11 discount levels 0-100%
Profit = np.zeros(11)

def State2Index(I):
    '''

    :param I: np.matrix. Contains product inventory for each step and age.
    :return: a four element array
    '''

    idx = I[0] # Retrieve the inventory of day 1.

    #loop over all days 1 to 4.
    for i in range(1, M):

        # the index value is updated by adding a value of 1 to
        # the upper boundary inventory number, and multiplying
        # this value with the previous index value. Additionally,
        # the existing values of age index 1 to 4 are added.
        # this is iteratred for M - 1 times.
        idx = (1 + UBI) * idx + I[i]
    return idx # a np.array of 4 elements is returned

def Index2State(idx):
    '''

    :param idx:
    :return:
    '''

    # initialize an empty inventory list
    Ilist = []

    # loop over all ages.
    for i in range(M):

        # create a new index by dividing the previous index
        # by 1 + the upper boundary inventory. This is a fraction
        # of how much product there is compared to what there is allowed
        # to be in stock
        idx1 = int(idx / (1 + UBI))

        # append the difference between the previous index and the
        # new index times 1 + upper boundary inventory.
        Ilist.append(idx - idx1 * (1 + UBI))
        print(idx, idx1, Ilist)
        idx = idx1
        return np.array(Ilist.reverse()) # return a list of new indices

## Evaluate fixed last day discounting by looping over different rates
for aa in range(11): # loop over discount percentages

    # re-initialize variables and random seeds.
    # set all their values back to 0.
    rd.seed(42)
    np.random.seed(42)
    I[:] = np.zeros([T + 1, M], dtype=int)
    IL[:] = np.zeros([T, M], dtype=int)
    BSPlvl[:] = np.zeros(T, dtype=int)
    Order[:] = np.zeros(T, dtype=int)
    D[:] = np.zeros(T, dtype=int)
    DF[:] = np.zeros(T, dtype=int)
    DL[:] = np.zeros(T, dtype=int)
    DeltaF[:] = np.zeros([T, M], dtype=int)
    DeltaL[:] = np.zeros([T, M], dtype=int)
    SALES[:] = np.zeros(T, dtype=int)
    PRF[:] = np.zeros(T)
    LS[:] = np.zeros(T, dtype=int)
    WST[:] = np.zeros(T, dtype=int)

    # we are still in the previous loop!
    # Simulate inventory dynamics over all periods.
    for t in range(T): # loop over all steps

        # observe state
        if t < 0: # if the step is negative, print the inventory of that step
            print(I[t], State2Index(I[t]))

        # predict demand and set Base Stock Policy level
        # (the target stock the policy wants to achieve)
        BSPlvl[t] = round(2 * mu + z * 2 * sqrt(mu))

        # set discount
        a = 0.05 * aa # discount level increments by a value of 0.05

        # set order quantity; this is either 0, or a value
        # that depends on the target stock - real stock.
        # if there is enough stock already, nothing is ordered.
        Order[t] = max(0, BSPlvl[t] - I[t].sum())

        # Set demand. This can be the total stock at maximum,
        # but can take a minimum of 0.
        # the real minimum value is randomly drawn from Poisson.
        D[t] = min(Dmx, np.random.poisson(mu))

        # Split demand. Check whether there are less products on
        # the last day or in total given the discount.
        DF_frac = min(I[t, M - 1], a * D[t])

        # if the discount was the previous minimal value,
        # this value is now changed into an integer.
        DF_int = int(DF_frac)

        # The Demand FEFO for this step is now updated.
        # hence, the demand for FEFO products is estimated for this step
        DF[t] = DF_int

        # Randomly increase the FEFO demand (fresh stock that customers
        # choose) in this step in case the expected demand was
        # calculated through the discounted demand formula above.
        if DF_frac != DF_int: # stochastic rounding

            # difference between rounded and fractioned FEFO demand
            Frac = DF_frac - DF_int

            # get random value between 0 and 1.
            u = rd.random()

            # the demand is increased by 1 if the random value is lower
            # than the FEFO fraction.
            if u < Frac:
                DF[t] = DF_int + 1

        # The total LEFO demand is the daily demand minus the FEFO demand
        # so either the full last day's inventory or part of it is excluded
        DL[t] = D[t] - DF[t]

        # pick FEFO demand:
        # only oldest fraction to sell are discounted (<= max I[t,M-1])
        # set the FEFO to sell
        DeltaF[t, M - 1] = DF[t]

        # available for LEFO customers/adjust stock levels:
        # LEFO inventory is calculated by subtracting today's FEFO demand
        # from the total inventory.
        IL[t] = I[t] - DeltaF[t]

        # pick LEFO demand:
        # add the youngest productg. This is either the daily LEFO demand or
        # LEFO inventory at day 1 if it is the smallest value.
        # you cannot sell more products in LEFO category if you did not order
        # that many the previous day
        DeltaL[t, 0] = min(DL[t], IL[t, 0])  # youngest first

        # for each each on this day, determine the LEFO demand change
        # that is either the daily LEFO demand minus the products that are
        # already freshly ordered, or the LEFO inventory of that day.
        for i in range(1, M):

            # simply set the new LEFO demand for each day
            # the demand for each older product decreases
            DeltaL[t, i] = min(DL[t] - DeltaL[t, 0:i].sum(), IL[t, i])

        # adjust stock levels:
        # the new stock levels for tomorrow's inventory is today's
        # LEFO inventory minus today's netto LEFO demand.
        # this is the inventory after both the FEFO and LEFO demand is satisfied
        I[t + 1, :] = IL[t] - DeltaL[t]

        # the waste of the oldest age products is the LEFO inventory space
        # minus the daily demand for LEFO products
        WST[t] = I[t + 1, M - 1]

        # tomorrow's inventory is today's inventory, but the supply
        # of the ages is moved to the next age. Hence, freshness 1
        # becomes freshness 2, and so on.
        I[t + 1, :] = np.roll(I[t + 1], 1)

        # the freshest product count tomorrow is what we had to order today
        I[t + 1, 0] = Order[t]  # as lead time is one period

        # more stats:
        # the lost sales because there was more demand; the daily demand
        # minus the total inventory space.
        LS[t] = max(0, D[t] - I[t].sum())

        # the number of units that were sold are the daily FEFO demand and
        # daily LEFO
        SALES[t] = DeltaF[t].sum() + DeltaL[t].sum()

        # calculate the profit in currency
        # LEFO profit, discounted FEFO profit minus order costs.
        PRF[t] = r * DeltaL[t].sum() + r * (1 - a) * DeltaF[t].sum() - c * Order[t]

    ## after one full episode:
    # COMPUTE & PRINT AVERAGES OF KEY PERFORMANCE MEASURES

    # average waste; what is the waste percentage of the total order?
    WasteRel = WST[Tw:T].mean() / Order[Tw:T].mean()

    # the fill rate is the 1 minus the average of how much product you
    # have lost compared to the average demand. So what percent of customers
    # on average could you not satisfy?
    Fillrate = 1 - LS[Tw:T].mean() / D[Tw:T].mean()

    # calculate the mean profit
    Profit[aa] = PRF[Tw:T].mean()

    # print these statistics for each discount value
    print(f"Discount = {a * 100:.0f}%, "
          f"Profit = {Profit[aa]:.3f}, "
          f"Waste = {WasteRel * 100:.1f} %, "
          f"Fillrate = {Fillrate * 100:.1f} %")




