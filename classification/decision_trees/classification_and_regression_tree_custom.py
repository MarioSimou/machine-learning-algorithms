import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

np.random.seed(1)

n = 10

x1_zeros = np.random.normal(2, 0.5, n)
x2_zeros = np.random.normal(2, 0.5, n)

x1_ones = np.random.normal(5, 1, n)
x2_ones = np.random.normal(5, 1, n)

# plt.scatter(x1_zeros, x2_zeros, label = 'zeros', c='r')
# plt.scatter(x1_ones, x2_ones, label = 'ones', c='b')

# plt.scatter(x1, x2)
# plt.hist(x1)
# plt.hist(x2)
# plt.legend(loc='best')
# plt.show()

x1 = list(x1_zeros) + list(x1_ones)
x2 = list(x2_zeros) + list(x2_ones)
y = [0 for _ in range(n)] + [1 for _ in range(n)]

dataset = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y,
})


def gini_cost_function(x1_size,x2_size):
    total = float(x1_size + x2_size)
    p1 = x1_size / total
    p2 = x2_size / total
    return 1 - (math.pow(p1, 2) + math.pow(p2, 2))

def gini_index(x1_size, x2_size, g00,g01,g10, g11):
    total = float(x1_size + x2_size)
    p1 = x1_size / total
    p2 = x2_size / total
    # g11 = 0
    # g12 = 0
    # g21 = 0
    # g22 = 0

    return ((1 - (math.pow(g00, 2) + math.pow(g01, 2))) * p1) + ((1 - (math.pow(g10, 2) + math.pow(g11, 2))) * p2)

# 0  2.812173  2.731054  0 - First split
# if x1 > 2.812173 RIGHT elif LEFT

# first 10 are left
# last 10 are right

v = x1[0] # best-split: 3

# y_est = []
# for x in x1:
#     if x > v:
#         y_est.append(1)
#     y_est.append(0)

total = float(len(x1))
y_est = list(map(lambda x: 1 if x > v else 0, x1))
n00 = len(list(filter(lambda x: True if x == 0 else False, y_est[:n])))
n01 = len(list(filter(lambda x: True if x == 1 else False, y_est[:n])))
g00 = n00 / n
g01 = n01 / n
n10 = len(list(filter(lambda x: True if x == 0 else False, y_est[n:])))
n11 = len(list(filter(lambda x: True if x == 1 else False, y_est[n:])))
g10 = n10 / n
g11 = n11 / n

print(gini_index(n, n, g00, g01, g10,g11))