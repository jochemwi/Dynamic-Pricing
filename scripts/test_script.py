from enviroment import Environment
from policy_iteration import PolicyIteration



# env = Environment(discount=0.01, discount_levels=51)
# print(env)

# state = env.reset()
# print(state)

# profits = []
# for discount_lvl in range(env.discount_levels):
#     # action = 0.05 * aa
#     action = env.discount * discount_lvl
#     # changed it to be dynamic :)
#     state = env.reset()
#     done = False
#     while not done:
#         next_state, reward, done = env.step(action)
#         state = next_state
#     stats = env.get_statistics()
#     profits.append(stats['waste'])

# env.plot_env_results(profits)


# state = env.reset()
# idx = env.state2index(state)
# print(idx)
# recovered = env.index2state(idx)
# print(recovered)

env = Environment(discount_levels=1, discount=0.0, max_shelf_life=5)
pi = PolicyIteration(env, gamma=1.0)
V, policy = pi.run()

print(f"Span of V: {V.max() - V.min():.2f}")
print(f"sim_length: {env.sim_length}")
print(f"Span / sim_length: {(V.max() - V.min()) / env.sim_length:.2f}")

# compare to simulation
env.reset()
for t in range(env.sim_length):
    env.step(0.0)
stats = env.get_statistics()
print(f"Simulation profit (no discount): {stats['profit']:.2f}")

