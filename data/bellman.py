import numpy as np

gamma = 0.9
n_states = 4
V = np.zeros(4)


def reward(s):
    if s==3:
        return 10
    return -1


for i in range(100):
    new_V = np.copy(V)
    for s in range(n_states):
        if s == 3:
            continue

        next_states = []

        if s > 0:
            next_states.append(s - 1)

        if s < n_states-1:
            next_states.append(s + 1)

        new_V[s] = max([reward(s) + gamma * V[s_state] for s_state in next_states])

    V = new_V

print(V)
