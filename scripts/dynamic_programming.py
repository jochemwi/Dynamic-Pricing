import numpy as np
import itertools as it
from numpy.typing import NDArray
from environment import Environment
import traceback

def get_variables(env:Environment):
    """Initiate all variables.

    :param env: Class Environment. Contains the environment.
    :return: state space, number of states, number of actions
        reward matrix and policy matrix.
    """
    bsp_lvl = env.base_stock
    s = np.arange(bsp_lvl + 1)
    ss = np.array(list(it.product(s, repeat = env.M)))

    valid = ss.sum(axis=1) <= bsp_lvl
    ss = ss[valid]
    ns = len(ss)

    na = env.discount_levels

    rr = np.zeros([na, ns])
    pp = np.zeros([na, ns, ns])
    return ss, ns, na, rr, pp

def check_states(pp:NDArray, action_states:bool = False) -> None:
    """Check number of possible states under the constraints of the environment.

    :param pp: 3D array of shape (na,ns,ns), contains the probabilities of
        reaching the next state given current state s and current action.
    :param action_states: bool, indicates whether only the number of
        possible states should be printed, or also the number of possible
        states while performing an action. The latter is only valid if the
        probability matrix is filled.
    :return: None
    """

    print(f"Total number of states: {pp.shape[1]}")
    for i in range(pp.shape[0]):
        count = 0
        for j in range(pp.shape[1]):
            if pp[i, j, :].sum() == 1.0:
                count += 1
        if action_states:
            print(f'Action {i} has {count} possible states.')


def get_matrices(env:Environment, ss, pp, rr):
    """Create probability and reward matrices.

    :param env: Class environment. Contains the environment.
    :param ss: np.array of shape (ns,M). Contains all states.
    :param pp: np.array of shape (na,ns,ns). Contains probabilities for
        next state space given a current state and action.
    :param rr: np.array of shape (na,ns). Contains the rewards for each
        action space.
    :return: rewards matrix and probability matrix.
    """

    for ida, a in enumerate(range(env.discount_levels)):
        env.reset()
        for ids, s in enumerate(ss):
            pp, tot_profit = env.get_probability_matrix(ss,pp,s,ida,ids,a)
            rr[ida, ids] = tot_profit
    return pp, rr

def policy_iteration_and_evaluation(na:int, ns:int, ss:NDArray, rr:NDArray, pp:NDArray, verbose:bool = False):
    """Perform policy iteration and iterative policy evaluation.

    :param na: int. Number of possible actions.
    :param ns: int. Number of possible states.
    :param ss: np.array of shape (ns,ns). Contains all state spaces.
    :param rr: np.array of shape (na,ns). Contains the rewards for action spaces.
    :param pp: np.array of shape (na, ns, ns). Contains probabilities of the next state
        given the current action and state.
    :param verbose: bool. Whether to track model progress while running. Default = False.
    :return: best policy, np.array of shape (ns).
    """

    pi = np.random.choice(np.arange(na), size = ns)
    v= np.zeros(ns)
    gamma = 0.999
    policy_not_stable = True
    n_iter_pi = 0

    while policy_not_stable:
        v, n_iter_ipe = iterative_policy_evaluation(ss, pi, na, v, rr, pp, gamma)
        policy_not_stable, pi = policy_improvement(pi,v,rr,pp,ns,na,ss)
        n_iter_pi += 1
        if verbose:
            print(f'n_iter_pi: {n_iter_pi}\tn_iter_ipe: {n_iter_ipe}')

    return pi

def iterative_policy_evaluation(ss:NDArray, pi:NDArray, na:int, v:NDArray,
                                 rr:NDArray, pp:NDArray, gamma:float = 0.9):
    """Perform iterative policy evaluation.

    :param ss: np.array of shape (ns,M). Contains all possible states.
    :param pi: np.array of shape (ns). Contains a policy.
    :param na: int. Number of possible actions.
    :param v: np.array of shape (ns). Contains the expected reward for
        any state.
    :param rr: np.array of shape (na,ns). Contains the rewards for
        action spaces.
    :param pp: np.array of shape (na, ns, ns). Contains probabilities of
        the next state given the current action and state.
    :param gamma: float. Discount in the Bellman equation. Default = 0.9.
    :return: Updated state value vector under the given policy and optionally
        the number of times the algorithm was run.
    """

    n_iter_ipe = 0
    theta = 0.0001 * (1 - gamma) / (2 * gamma)
    span = 1 + theta
    while span > theta:
        v_prev = v.copy()
        n_iter_ipe += 1
        for ids, s in enumerate(ss):
            a = pi[ids]
            ida = np.where(np.arange(na) == a)[0][0]
            v[ids] = rr[ida, ids] + gamma * pp[ida, ids, :] @ v_prev

        diff = v - v_prev
        span = max(diff) - min(diff)
        n_iter_ipe += 1
    return v, n_iter_ipe

def policy_improvement(pi:NDArray, v:NDArray, rr:NDArray, pp:NDArray,
                       ns:int, na:int, ss:NDArray, gamma:float = 0.9):
    """Perform policy improvement.

    :param pi: np.array of shape (ns). Contains a policy.
    :param v: np.array of shape (ns). Contains the expected state values
        for any state.
    :param rr: np.array of shape (na,ns). Contains the rewards for
        action spaces.
    :param pp: np.array of shape (na, ns, ns). Contains probabilities of
        the next state given the current action and state.
    :param ns: int. Number of states.
    :param na: int. Number of actions.
    :param ss: np.array of shape (ns, ns). Contains all states.
    :param gamma: float. Discount according to the Bellman equation.
        Default = 0.9.
    :return: bool, whether the best policy was found. np.array of shape (ns)
        that contains the best policy.
    """

    q = np.zeros([na, ns])

    pi_prev= pi.copy()
    policy_not_stable = False
    for ids, s in enumerate(ss):
        q_best = - np.inf
        for ida, a in enumerate(np.arange(na)):
            q[ida, ids] = rr[ida, ids] + gamma * pp[ida, ids, :] @ v
            if q[ida, ids] > q_best:
                pi[ids] = a
                q_best = q[ida, ids]
        if pi[ids] != pi_prev[ids]:
            policy_not_stable = True
    return policy_not_stable, pi

def print_policy_contents(pi:NDArray) -> None:
    """Print the number of times an action is chosen in pi.

    :param pi: np.array of shape (ns). Contains the action to take
        for a given state.
    :return: None
    """
    val_d = {}
    for i in pi:
        if i in val_d:
            val_d[i] += 1
        else:
            val_d[i] = 1

    for key in sorted(val_d.keys()):
        print(f'Action {key} is chosen {val_d[key]} times')

def run_dp_model(env:Environment, pi:NDArray, ss:NDArray):
    """Run the best policy in the environment.

    :param env: Class environment. Contains the environment.
    :param pi: np.array of shape (ns). Contains a policy (the best).
    :param ss: np.array of shape (ns,ns). Contains the all states.
    :return: mean profit, waste and fill rate, all float.
    """
    env.reset()

    done = False
    while not done:
        pi_a_idx = np.where((ss == env.inventory_matrix[env.t])
                            .all(axis = 1))[0][0].item()
        action = pi[pi_a_idx]
        _, _, done = env.step(action)

    profit = env.get_statistics()['profit']
    waste = env.get_statistics()['waste']
    fill_rate = env.get_statistics()['fill_rate']
    return profit, waste, fill_rate

def value_iteration(na:int, ns:int, ss:NDArray, rr:NDArray, pp:NDArray,
                    gamma:float = 0.9, k:int = 50, theta:int = 1,
                    verbose = False):
    """Perform value iteration as dynamic programming.

    :param na: int. Number of actions.
    :param ns: int. Number of states.
    :param ss: np.array of shape (ns,ns). Contains all states.
    :param rr: np.array of shape (na,ns). Contains all rewards given a state
        action.
    :param pp: np.array of shape (na,ns,ns). Contains all probabilities for
        a next state given the current state and action.
    :param gamma: float. The discount value in the Bellman equation.
        Default = 0.9.
    :param k: int. maximum number value Iteration has to be performed.
        Default = 50.
    :param theta: int. Limits the model to perform value Iteration when
        the difference between new and old state value vectors is nihil.
            Default = 1.
    :param verbose: bool. Whether algorithm progress should be visible.
        Default = False.
    :return: best policy of type np.array with shape (ns)
    """

    pi = np.random.choice(np.arange(na), ns)
    v= np.zeros(ns)
    n_iter_vi = 0
    stop = False
    kk = 0

    while not stop:
        if verbose:
            print(f'n_iter_vi: {n_iter_vi}')
        kk += 1
        delta = 0

        for ids, s in enumerate(ss):
            vs = v[ids]
            v_best = -99999

            for ida, a in enumerate(np.arange(na)):
                v_try = rr[ida,ids] + gamma * pp[ida, ids, :] @ v
                if v_try > v_best:
                    v_best = v_try
                    pi[ids] = a
            v[ids] = v_best
            if delta < abs(v[ids] - vs):
                delta = abs(v[ids] - vs)
        if (kk == k) or (delta < theta):
            stop = True
        n_iter_vi += 1

    return pi

def run_ipe_and_pi(env:Environment):
    """Run Iterative Policy Evaluation and Policy Improvement.

    :param env: Class Environment. Contains the environment.
    :return: policy, profit, waste, fill rate
    """

    print('=====PROGRESS IPE+PI=====')
    print('0%')
    ss, ns, na, rr, pp = get_variables(env)
    print('20%')
    pp, rr = get_matrices(env,ss,pp,rr)
    print('50%')
    pi = policy_iteration_and_evaluation(na,ns,ss,rr,pp)
    print('70%')
    profit, waste, fill_rate = run_dp_model(env, pi, ss)
    print('100')
    print('=====END IPE+PI=====\n')
    return pi, profit, waste, fill_rate

def run_vi(env:Environment):
    """Run value Iteration.

    :param env: Class Environment. Contains the environment.
    :return: policy, profit, waste, fill rate
    """

    print('=====PROGRESS IPE+PI=====')
    print('0%')
    ss, ns, na, rr, pp = get_variables(env)
    print('20%')
    pp, rr = get_matrices(env,ss,pp,rr)
    print('40%')
    pi = value_iteration(na,ns,ss,rr,pp)
    print('70%')
    profit, waste, fill_rate = run_dp_model(env, pi, ss)
    print('100%')
    print('=====END IPE+PI=====\n')
    return pi, profit, waste, fill_rate