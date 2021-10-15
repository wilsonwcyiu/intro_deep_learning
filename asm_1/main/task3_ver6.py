import pprint
from random import random

import matplotlib
import numpy as np
# import h5py
import matplotlib.pyplot as plt
# from testCases_v4a import *
# from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


# Implement the function xor net(x1; x2;weights) that simulates a network with
# two inputs, two hidden nodes and one output node.
# The vector weights denotes 9 weights (tunable parameters):
# each non-input node has three incoming weights:
# one connected to the bias node that has value 1, and
# two other connections that are leading from the input nodes to a hidden node or from the two hidden nodes to the output node.
# Assume that all non-input nodes use the sigmoid activation function.


def sigmoid(z: np.array):
    return 1/(1 + np.exp(-z))


def mse_derivative(activation: np.array):
    return sigmoid(activation) * (1 - sigmoid(activation))


def mse(weights_1: list, weight2: list):
    n = len(weights_1)
    sum: float = 0
    for idx in range(0, n):
        sum += (weights_1[idx] - weight2[idx])**2

    mse: float = sum/n

    return mse



def grdmse(cost: float):
    mse_der: float = mse_derivative(cost)
    weight_vector_grd: list = [mse_der, mse_der, mse_der,
                               mse_der, mse_der, mse_der,
                               mse_der, mse_der, mse_der]

    return weight_vector_grd



def xor_net(x1_list, x2_list, weight_vector: list):
    yhat_list: list = []
    for idx in range(0, len(x1_list)):
        x1 = x1_list[idx]
        x2 = x2_list[idx]

        # feedforward
        neural_1_z: float = fixed_bias_1 * weight_vector[0] + x1 * weight_vector[1] + x2 * weight_vector[2]
        neural_1_activation: float = sigmoid(neural_1_z)

        neural_2_z: float = fixed_bias_1 * weight_vector[3] + x1 * weight_vector[4] + x2 * weight_vector[5]
        neural_2_activation: float = sigmoid(neural_2_z)

        output_neural_3_z: float = fixed_bias_1 * weight_vector[6] + neural_1_activation * weight_vector[7] + neural_2_activation * weight_vector[8]
        output_activation: float = sigmoid(output_neural_3_z)

        yhat_list.append(output_activation)

    return yhat_list





if __name__ == '__main__':


    x1: int = [0, 0, 1, 1]
    x2: int = [0, 1, 1, 0]
    all_y_label_list: list = [0, 1, 1, 0]

    fixed_bias_1: int = 1

    np.random.seed(42)
    weight_vector: list = np.random.rand(9, 1)

    lr = 0.001
    idx = 0
    for epoch in range(200000):
        yhat_list = xor_net(x1, x2, weight_vector)

        loss_list: float = []
        for loss_idx in range(0, len(yhat_list)):
            loss: float = yhat_list[loss_idx] - all_y_label_list[loss_idx];
            loss_list.append(loss**2)

        cost = np.sum(loss_list)/ len(loss_list);               print(cost)

        gradient_desc_vector = grdmse(cost)

        weight_vector[0] -= lr * gradient_desc_vector[0]
        weight_vector[1] -= lr * gradient_desc_vector[1]
        weight_vector[2] -= lr * gradient_desc_vector[2]
        weight_vector[3] -= lr * gradient_desc_vector[3]
        weight_vector[4] -= lr * gradient_desc_vector[4]
        weight_vector[5] -= lr * gradient_desc_vector[5]
        weight_vector[6] -= lr * gradient_desc_vector[6]
        weight_vector[7] -= lr * gradient_desc_vector[7]
        weight_vector[8] -= lr * gradient_desc_vector[8]



