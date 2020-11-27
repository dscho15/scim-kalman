import numpy as np
import matplotlib
import matplotlib.pyplot as plt

t = np.linspace(0, 2*np.pi, num = 2000)

def sin_4(A, omega, t):
    position     = A*(1-np.cos(omega*t))
    velocity     = -A*(omega)**1*np.sin(omega*t)
    acceleration = -A*(omega)**2*np.cos(omega*t)
    jerk         = A*(omega)**3*np.sin(omega*t)
    return position, velocity, acceleration, jerk


# 22.5 degrees
A = 0.196349541

for i in range(1, 4):

    pos, vel, acc, jerk = sin_4(A, 2*np.pi*0.375*i, t)

    plt.subplot(4, 1, 1)
    plt.plot(t, pos)

    plt.subplot(4, 1, 2)
    plt.plot(t, vel)

    plt.subplot(4, 1, 3)
    plt.plot(t, acc)

    plt.subplot(4, 1, 4)
    plt.plot(t, jerk)

plt.show()