import pandas as pd
from matplotlib import pyplot as plt
from functools import reduce

x = [1,2,3,4,5]
y = [1,3,2,3,5]

dataset = pd.DataFrame({
    'x': x,
    'y': y
})


# print(dataset.head())

# print(dataset.info())


# dataset.plot(kind='scatter', x='x',y='y')
# plt.show()

# Simple Linear Regression
# y = b0 + b1 * x


def mean(row):
    return reduce(lambda s, x: s + x, row, 0) / len(row)

def calculate_intercept(x,y):
    # mean(y) - mean(x) * b1

    b1 = calculate_coefficient(x,y)
    return mean(y) - mean(x) * b1

def calculate_coefficient(x,y):
    # sum((xi - mean(x)) * (yi - mean(y))) / sum(power(xi - mean(x), 2))
    if len(x) != len(y):
        raise Exception('X and Y dataset not the same size')
    
    xm = mean(x)
    ym = mean(y)

    dataset = zip(x,y)
    b1 = reduce(lambda s, d: s + (d[0]-xm) * (d[1]-ym), dataset, 0) / reduce(lambda s, x: s + (x - xm) ** 2, x, 0)
    return b1


def LinearRegression(x,y):
    b0 = calculate_intercept(x,y)
    b1 = calculate_coefficient(x,y)

    def predict(*x):
        return list(map(lambda x: b0 + b1 * x, x) )

    return predict


model = LinearRegression(x,y)
x_test = [1,2,4,3,5]
y_pre = model(*x_test)
print(y_pre)

def RMSE(y, y_pre):
    # sqrt(sum(power(y_hat_i - yi, 2)) / n)
    dataset = zip(y,y_pre)
    return (reduce(lambda s,d: s + (d[1] - d[0]) ** 2, dataset, 0) / len(y)) ** 0.5 

# plt.scatter(x,y,c='b', marker='o')
# plt.plot(x_test, y_pre, c='r')
# plt.show()

print(f'RMSE: {RMSE(y,y_pre)}')