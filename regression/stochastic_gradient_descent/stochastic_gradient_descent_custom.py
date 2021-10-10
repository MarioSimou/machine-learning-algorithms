from functools import reduce
from math import inf
from matplotlib import pyplot as plt

x = [1,2,4,3,5]
y = [1,3,3,2,5]

def training_model(aplha = 0.1):
    global b0 
    global b1

    b0 = 0
    b1 = 0

    def model(xi):
        return b0 + b1 * xi

    def predict(xi,yi):
        global b0
        global b1

        y_pre = model(xi)
        e = y_pre - yi
 
        b0 = b0 - (aplha * e)
        b1 = b1 - (aplha * e * xi)

        return (y_pre, e)

    return predict

def StochasticGradientDescent(x,y, e_acc = 0.1, max_iter = 1000, aplha = 0.01):
    if len(x) != len(y):
        raise Exception('X and Y sets should be the same length')

    n = len(x)
    e = inf
    cur_iter = 0
    predict = training_model(aplha)
    i = 0
    # e_list = []
    # cur_iter_list = []

    while e > e_acc or cur_iter < max_iter:
        xi = x[i]
        yi = y[i]

        _, e_cur = predict(xi,yi)
        # e_list.append(e)
        # cur_iter_list.append(cur_iter)

        if abs(e_cur) < abs(e_acc) and cur_iter > 0:
            break

        e = e_cur

        if i < n-1:
            i = i + 1
        else:
            i = 0

        cur_iter = cur_iter + 1

    # print(list(zip(cur_iter_list, e_list)))
    # plt.plot(cur_iter_list, e_list)
    # plt.show()
    return predict

StochasticGradientDescent(x,y, 0.1, 100, 0.01)