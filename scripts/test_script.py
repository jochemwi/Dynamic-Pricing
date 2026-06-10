from enviroment import Environment



env = Environment(discount_levels=22)
print(env)

state = env.reset()
print(state)

profits = []
for aa in range(env.discount_levels):
    # action = 0.05 * aa
    action = aa / (env.discount_levels - 1) # changed it to be dynamic :)
    state = env.reset()
    done = False
    while not done:
        next_state, reward, done = env.step(action)
        state = next_state
    stats = env.get_statistics()
    profits.append(stats['profit'])

env.plot_env_results(profits)


state = env.reset()
idx = env.state2index(state)
print(idx)
recovered = env.index2state(idx)
print(recovered)