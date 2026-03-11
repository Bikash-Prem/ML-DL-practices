import numpy as np
import matplotlib.pyplot as plt

g = 9.81
L = 1.0
dt = 0.1
T = 5

theta = 0.1
theta_dot = 0.0

theta_list = []
time_list = []

t = 0

while t < T:
    kp = 20
    kd = 5
    u = -kp*theta - kd*theta_dot

    disturbance = np.random.normal(0,1)

    theta_ddot = (g/L) * np.sin(theta) + u + disturbance
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    theta_list.append(theta)
    time_list.append(t)

    t += dt

plt.plot(time_list, theta_list)
plt.xlabel("time(s)")
plt.ylabel("angle(rad)")
plt.title("Inverted Pendulum")
plt.show()
