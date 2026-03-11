import numpy as np
import random
import matplotlib.pyplot as plt


class LineWorld:
    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, actions):
        self.state += actions

        if self.state == 2:
            return self.state, 1, True

        elif self.state == -2:
            return self.state, -1, True

        else:
            return self.state, 0, False


states = [-1, 0, 1]
actions = [-1, 1]

Q = {}

for s in states:
    for a in actions:
        Q[(s, a)] = 0.0

alpha = 0.1  # Learning Rate
gamma = 0.9  # discount factor
epsilon = 0.2  # exploration rate
episodes = 500

env = LineWorld()
episode_reward = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = max(actions, key=lambda a: Q[(state, a)])

        next_state, reward, done = env.step(action)

        if next_state in [-2, 2]:
            max_future = 0
        else:
            max_future = max(Q[(next_state, a)] for a in actions)

        # Q-Learning Update
        Q[(state, action)] += alpha*(
            reward + gamma * max_future - Q[(state, action)]
        )

        state = next_state
        total_reward += reward
    episode_reward.append(total_reward)


plt.plot(episode_reward)
plt.xlabel("Episode")
plt.ylabel("Total Learning")
plt.title("Learning Progress")
plt.show()

print("---Learning Policy---")
for s in states:
    best_action = max(actions, key = lambda a: Q[(s, a)])
    print(f"State {s} -> Best Action {best_action}")

print("---Final Table---")
for key in Q:
    print(key, ":", round(Q[key], 3))

