import math
from matplotlib import pyplot as plt
from functools import reduce
x1 = [
    2.781026,
    1.465489,
    3.396561,
    1.388070,
    3.064072,
    7.627531,
    5.332441,
    6.922596,
    8.675418,
    7.673756
]

x2 = [
    2.550537,
    2.362125,
    4.400293,
    1.850220,
    3.005305,
    2.759262,
    2.088626,
    1.771063,
    -0.242068,
    3.508563,
]

y = [0,0,0,0,0,1,1,1,1,1]


def logistic_regression(x1, x2, y, alpha = 0.1, max_epochs = 10, min_acc = 0.8):
    global b0
    global b1
    global b2

    b0 = 0
    b1 = 0
    b2 = 0

    def logistic_fn(model , *args):
        return 1 / (1 + (1/math.exp(model(*args))))
    
    def linear_model(x1i, x2i):
        return b0 + b1 * x1i + b2 * x2i

    def train_data_point(x1i, x2i, yi):
        global b0
        global b1
        global b2

        y_pre = logistic_fn(linear_model, x1i, x2i)
        b0 = b0 + alpha * (yi - y_pre) * y_pre * (1 - y_pre)
        b1 = b1 + alpha * (yi - y_pre) * y_pre * (1 - y_pre) * x1i
        b2 = b2 + alpha * (yi - y_pre) * y_pre * (1 - y_pre) * x2i

        return y_pre

    def get_class(y_pre):
        if y_pre > 0.5:
            return 1
        else:
            return 0

    def predict(x1i, x2i):
        return get_class(logistic_fn(linear_model, x1i, x2i))

    def train(x1, x2, y):
        dataset = list(zip(x1,x2,y))
        cur_epoch = 0
        accuracies_list = []

        while cur_epoch < max_epochs:
            y_pre_list = []
            for x1i, x2i, yi in dataset:
                y_pre = train_data_point(x1i, x2i, yi)
                y_pre_list.append(get_class(y_pre))
            
            cur_acc = reduce(lambda matches,predictions: matches + 1 if predictions[0] == predictions[1] else matches, list(zip(y_pre_list, y)), 0) / len(y_pre_list)
            accuracies_list.append(cur_acc)            

            if cur_acc > min_acc:
                break
            
            cur_epoch = cur_epoch + 1

        # print(accuracies_list)
        # plt.plot(list(range(1,11)), accuracies_list)
        # plt.show()
        return predict

    return train(x1,x2,y)


predict = logistic_regression(x1, x2, y, 0.3, 10, 1)

zero_set_x1 = x1[:5]
zero_set_x2 = x2[:5]
one_set_x1 = x1[5:]
one_set_x2 = x2[5:]

plt.scatter(zero_set_x1, zero_set_x2, c='red', marker='o', label = 'zeros')
plt.scatter(one_set_x1, one_set_x2, c='blue', marker='^', label = 'ones')
plt.legend(loc='best')

x1_test = list(range(1,11))
x2_test = list(range(0,6))
y_predictions = [predict(x1i,x2i) for x1i,x2i in zip(x1,x2_test)]

p = sorted(zip(y_predictions,x1_test, x2_test))
zeros = p[:3]
ones = p[3:]

plt.plot([x1i for (yi,x1i,x2i) in zeros], [x2i for (yi,x1i,x2i) in zeros])
plt.plot([x1i for (yi,x1i,x2i) in ones], [x2i for (yi,x1i,x2i) in ones])

plt.show()