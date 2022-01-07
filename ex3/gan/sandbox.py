import random

from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    list1 = list(range(100))
    random_noise = [random.randint(0, 20) for i in range(100)]
    list2 = [a + b for a, b in zip(list1, random_noise)]
    plt.plot(list2)
    plt.show()
    n = 10
    moving_averages = [np.mean(list2[i - n: i + n]) for i in range(n, len(list2) - n)]
    plt.plot(moving_averages)
    plt.show()
