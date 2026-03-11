import numpy as np


def reward(s):
    if s == 3:
        return 10
    return -1


Q = np.zeros((4, 2))  # states x actions

alpha = 0.1
gamma = 0.9

for episode in range(1000):
    s = 0

    while s != 3:
        a = np.random.choice([0, 1])  # explore

        # deterministic transition
        s_next = max(0, min(3, s + (1 if a == 1 else -1)))
        r = reward(s_next)

        Q[s, a] += alpha * (
                r + gamma * np.max(Q[s_next]) - Q[s, a]
        )

        s = s_next

print(Q)