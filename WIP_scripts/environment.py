class Environment():
    def __init__(
            self, 
            max_shelf_life=5,           
            mu=3,                       
            regular_sales_price=250,   
            purchase_price=175,
            discount_levels=11,
            discount = 0.05,
            warm_up = 10,
            z = 1/sqrt(3),
            seed = 42,
            ):
        '''Initialise the perishable inventory environment.

        Sets up demand distribution, base stock level, and the upper bound
        on inventory per age slot (UBI), then resets all tracking arrays
        via self.reset().

        :param max_shelf_life: number of days before an item expires (M)
        :param mu: mean daily demand, used for the Poisson demand distribution
        :param regular_sales_price: selling price per item without discount
        :param purchase_price: cost per item ordered
        :param discount_levels: number of discrete discount levels (0% to 50%)
        :param discount: step size between discount levels (e.g. 0.05 = 5% per level)
        :param warm_up: number of initial periods excluded from performance statistics
        :param z: safety factor used to compute the base stock level and UBI
        :param seed: random seed for reproducibility (numpy and random module)

        :return: None
        '''
        
        self.M = max_shelf_life                         
        self.mu = mu                                    
        self.max_demand = mu * 3                        # maximise demand distribution at 3x mean
        self.regular_sales_price = regular_sales_price
        self.purchase_price = purchase_price
        self.discount_levels = discount_levels
        self.discount = discount          
        self.safety_factor = z
        self.warm_up = warm_up
        self.sim_length = self.warm_up + 36500          # total simulation length including warm-up
        self.prob = self.probability()                  # demand probabilities following Poisson
        self.UBI = int(min(self.max_demand, 2*self.mu + self.safety_factor * sqrt(2)*self.mu))
        self.base_stock = round(2*self.mu + self.safety_factor * sqrt(2)*self.mu)
        self.seed = seed
        np.random.seed(seed)
        rd.seed(seed)

        self.reset()                                    # initialise arrays on initialisation
    
    def probability(self):
        '''Compute truncated Poisson demand probabilities.

        Returns the probability mass function for daily demand, truncated
        at self.max_demand. The final entry absorbs all remaining
        probability mass so the distribution sums to 1.

        :return: probability mass for demand values 0 to max_demand
            type: np.array with shape (max_demand + 1,)
        '''
        prob = poisson.pmf(np.arange(0, self.max_demand + 1), self.mu)
        prob[self.max_demand] = 1 - prob[0:self.max_demand].sum()
        return prob

    def reset(self):
        '''Initialise all tracking arrays and return the initial state.

        Allocates fresh zero-filled arrays for inventory and all per-period
        statistics, sized to self.sim_length (or sim_length+1 for inventory_matrix,
        which also stores the state after the final period). Resets the time
        step counter to 0.

        :return: index of the initial state (all-zero inventory)
            type: int
        '''
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
        return self.state2index(self.inventory_matrix[0])                                   
    
    def split_demand(self, action):
        '''Split total demand into FEFO (discounted) and LEFO (full-price) demand.

        FEFO customers are attracted to the oldest stock by the discount rate
        (action). The number of FEFO customers is capped at the available
        oldest-age inventory (slot M-1). Fractional demand is resolved via
        stochastic rounding. Remaining demand is assigned to LEFO customers,
        who buy fresh stock.

        :param action: discount rate applied to the oldest stock (fraction of demand attracted to discount)
        :return: None
        '''
        # action * demand[t] = fraction of demand attracted to discounted (oldest) items,
        # capped by available oldest stock inventory_matrix[t, M-1]
        self.fefo_demand_frac = min(self.inventory_matrix[self.t, self.M-1], action * self.demand[self.t])
        self.fefo_demand_int = int(self.fefo_demand_frac)
        self.fefo_demand[self.t] = self.fefo_demand_int

        # stochastic rounding: round up with probability equal to the fractional part
        if self.fefo_demand_frac != self.fefo_demand_int:
            frac = self.fefo_demand_frac - self.fefo_demand_int
            if rd.random() < frac:
                self.fefo_demand[self.t] = self.fefo_demand_int + 1

        # remaining demand goes to LEFO customers (buy fresh stock)
        self.lefo_demand[self.t] = self.demand[self.t] - self.fefo_demand[self.t]

    def update_inventory(self):
        '''Update inventory after FEFO and LEFO picks, then age stock by one day.

        FEFO customers take only from the oldest age class. Remaining LEFO
        demand is met starting from the freshest stock, spilling into older
        slots as needed. Items left in the oldest slot expire as waste before
        the inventory ages by one day and the new order arrives in slot 0.

        :return: None
        '''
        # FEFO customers only pick from the oldest age class (M-1)
        self.items_picked_fefo[self.t, self.M-1] = self.fefo_demand[self.t]

        # stock remaining after FEFO picks, available to LEFO customers
        self.inventory_after_fefo[self.t] = self.inventory_matrix[self.t] - self.items_picked_fefo[self.t]

        # LEFO customers buy freshest first (age 0), capped by available stock
        self.items_picked_lefo[self.t, 0] = min(self.lefo_demand[self.t], self.inventory_after_fefo[self.t, 0])

        # if slot 0 didn't cover full LEFO demand, spill into progressively older slots
        for i in range(1, self.M):
            self.items_picked_lefo[self.t, i] = min(
                self.lefo_demand[self.t] - self.items_picked_lefo[self.t, 0:i].sum(),
                self.inventory_after_fefo[self.t, i]
            )

        # next inventory = remaining stock after all picks (FEFO and LEFO)
        self.inventory_matrix[self.t+1, :] = self.inventory_after_fefo[self.t] - self.items_picked_lefo[self.t]

        # waste = oldest items still unsold; these expire at the end of this period
        self.waste[self.t] = self.inventory_matrix[self.t+1, self.M-1]

        # age all items by one day (shift columns right); oldest wraps to slot 0,
        # but gets overwritten by the new order on the next line
        self.inventory_matrix[self.t+1, :] = np.roll(self.inventory_matrix[self.t+1], 1)

        # new order arrives in slot 0 (lead time = 1 period)
        self.inventory_matrix[self.t+1, 0] = self.order_quantity[self.t]

    def compute_stats(self, action, p=None):
        '''Compute lost sales, sales, and profit for the current period.

        Used both by the simulation step (RL methods, p=None) and by DP's
        reward matrix construction, where p is the transition probability
        used to weight profit into an expected reward contribution.

        :param action: discount rate applied this period (used to discount FEFO sales revenue)
        :param p: transition probability to weight profit by, for DP reward calculation.
            If None, profit is left unweighted (used during simulation).
        :return: None
        '''
        # lost sales = demand that could not be met from available stock
        self.lost_sales[self.t] = max(0, self.demand[self.t] - self.inventory_matrix[self.t].sum())

        # total sales = all items picked by both customer types (FEFO + LEFO)
        self.sales[self.t] = self.items_picked_fefo[self.t].sum() + self.items_picked_lefo[self.t].sum()

        # profit = regular sales revenue + discounted sales revenue - purchase cost
        self.profit[self.t] = (self.regular_sales_price * self.items_picked_lefo[self.t].sum() \
                        + self.regular_sales_price * (1 - action) * self.items_picked_fefo[self.t].sum() \
                        - self.purchase_price * self.order_quantity[self.t])

        if p is not None:
            self.profit[self.t] = self.profit[self.t] * p

    def get_statistics(self):
        '''Compute summary performance metrics, excluding the warm-up period.

        :return: dictionary with average profit, relative waste, and fill rate
            type: dict with keys 'profit', 'waste', 'fill_rate'
        '''
        waste_rel = self.waste[self.warm_up:].mean() / self.order_quantity[self.warm_up:].mean()
        fill_rate = 1 - self.lost_sales[self.warm_up:].mean() / self.demand[self.warm_up:].mean()
        profit = self.profit[self.warm_up:].mean()
        return {
            'profit': profit,
            'waste': waste_rel,
            'fill_rate': fill_rate
        }

    def step(self, action):
        '''Advance the environment by one period using the given action.

        Computes the order quantity (up to the base stock level), draws
        demand, splits it between FEFO and LEFO customers, updates inventory,
        and computes the resulting profit. Advances the time step and
        signals whether the simulation horizon has been reached.

        :param action: discrete action index (0 to discount_levels-1),
            converted to a discount fraction via action * self.discount
        :return: tuple of (next_state_idx, reward, done)
            next_state_idx: int, index of the resulting inventory state
            reward: float, profit earned this period
            done: bool, True if the simulation horizon has been reached
        '''
        # convert discrete action integer to discount fraction, e.g. action=3 → 0.15 → 15% discount
        discount_fraction = action * self.discount

        self.base_stock_level[self.t] = round(2*self.mu + self.safety_factor * sqrt(2)*self.mu)
        self.order_quantity[self.t] = max(0, self.base_stock_level[self.t] - self.inventory_matrix[self.t].sum())
        self.demand[self.t] = min(self.max_demand, np.random.poisson(self.mu))

        self.split_demand(discount_fraction)
        self.update_inventory()
        self.compute_stats(discount_fraction)

        reward = self.profit[self.t]
        next_state_vec = self.inventory_matrix[self.t + 1]
        next_state_idx = self.state2index(next_state_vec)

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
        '''Compute expected profit and fill in transition probabilities for a given state-action pair.

        Loops over all possible demand realizations, simulating the resulting
        next state for each. Accumulates the transition probability into PP
        and computes the demand-weighted expected profit for this (state, action) pair.

        :param SS: np.array of shape (nS, M). All possible inventory states.
        :param PP: np.array of shape (nA, nS, nS). Transition probabilities;
            updated in place for this (ida, ids) pair across all next states.
        :param ida: int. Index of the current action.
        :param ids: int. Index of the current state.
        :param discount_fraction: float. Discount rate corresponding to the current action.
        :return: tuple (PP, tot_profit)
            PP: np.array of shape (nA, nS, nS), updated transition probability matrix
            tot_profit: float, expected profit for this (ids, ida) pair, summed over all demand realizations weighted by their probability
        '''
        
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
        '''Plot profit against discount percentage for fixed last-day discounting.

        :param profit: array-like, profit value for each discount level
        :return: None
        '''
        step = self.discount * 100
        plt.plot(step * np.arange(self.discount_levels), profit,  marker='o')
        plt.title("Fixed last-day discounting")
        plt.xlabel("discount %")
        plt.ylabel("profit")
        plt.grid(True)
        plt.show()

    def __repr__(self):
        '''Return a readable summary of the environment's configuration.

        :return: string representation showing key parameters
            type: str
        '''
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
