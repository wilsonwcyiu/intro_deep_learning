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


def sigmoid(z: float):
    return float(1/(1 + np.exp(-z)))


def sigmoid_derivative(activation: np.array):
    return sigmoid(activation) * (1 - sigmoid(activation))


def mse(weights_1: list, weight2: list):
    n = len(weights_1)
    sum: float = 0
    for idx in range(0, n):
        sum += (weights_1[idx] - weight2[idx])**2

    mse: float = sum/n

    return mse



def grdmse(lost: float, yhat: float, activation_array: np.array):
    # output layer to hidden layer grd
    dcost_dpred: np.array = lost;
    # dpred_dz: np.array = sigmoid_derivative([activation_array[6], activation_array[7], activation_array[8]]);

    # grdmse_activation_delta: np.array = dcost_dpred * dpred_dz;                             #print("z_delta.shape", z_delta.shape)

    weight_vector_grd: list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    weight_vector_grd[6] = dcost_dpred * sigmoid_derivative(yhat)
    weight_vector_grd[7] = dcost_dpred * sigmoid_derivative(yhat)
    weight_vector_grd[8] = dcost_dpred * sigmoid_derivative(yhat)


    # hidden layer to input layer grd
    # neural_1_lost: float = activation_array[7] -
    # dcost_dpred: np.array = lost;
    # # dpred_dz: np.array = , activation_array[7], activation_array[8]]);
    #
    # weight_vector_grd[6] = dcost_dpred * sigmoid_derivative(activation_array[6])
    # weight_vector_grd[7] = dcost_dpred * sigmoid_derivative(activation_array[7])
    # weight_vector_grd[8] = dcost_dpred * sigmoid_derivative(activation_array[8])
    weight_vector_grd[0] = dcost_dpred * sigmoid_derivative(activation_array[7])
    weight_vector_grd[1] = dcost_dpred * sigmoid_derivative(activation_array[8])
    weight_vector_grd[2] = dcost_dpred * sigmoid_derivative(activation_array[7])
    weight_vector_grd[3] = dcost_dpred * sigmoid_derivative(activation_array[8])
    weight_vector_grd[4] = dcost_dpred * sigmoid_derivative(activation_array[7])
    weight_vector_grd[5] = dcost_dpred * sigmoid_derivative(activation_array[8])

    return np.array(weight_vector_grd)



def xor_net(x1: int, x2: int, weight_vector: list):
    # feedforward
    neural_1_z: float = fixed_bias_1 * weight_vector[0] + x1 * weight_vector[1] + x2 * weight_vector[2]
    neural_1_activation: float = sigmoid(neural_1_z)

    neural_2_z: float = fixed_bias_1 * weight_vector[3] + x1 * weight_vector[4] + x2 * weight_vector[5]
    neural_2_activation: float = sigmoid(neural_2_z)

    output_neural_3_z: float = fixed_bias_1 * weight_vector[6] + neural_1_activation * weight_vector[7] + neural_2_activation * weight_vector[8]
    output_activation: float = sigmoid(output_neural_3_z)

    return output_activation





if __name__ == '__main__':



    all_train_list: list = [[0,0],[0,1],[1,0],[1,1]]
    all_y_label_list: list = [0, 1, 1, 0]

    fixed_bias_1: int = 1

    np.random.seed(42)
    weight_vector: list = np.random.rand(9, 1)

    lr = 0.001
    idx = 0
    for epoch in range(200000):

        train_list = all_train_list[idx]
        x1: int = train_list[0]
        x2: int = train_list[1]

        y_result: int = all_y_label_list[idx]


        loss_list: float = []
        for case in range(0,4):
            idx += 1
            if idx == 4:
                idx = 0


            yhat: float = xor_net(x1, x2, weight_vector)

            loss: float = yhat - y_result;
            loss_list.append(loss**2)

        cost = np.sum(loss_list)/ len(loss_list);               print(cost)

        neural_1_z: float = fixed_bias_1 * weight_vector[0] + x1 * weight_vector[1] + x2 * weight_vector[2]
        neural_1_activation: float = sigmoid(neural_1_z)

        neural_2_z: float = fixed_bias_1 * weight_vector[3] + x1 * weight_vector[4] + x2 * weight_vector[5]
        neural_2_activation: float = sigmoid(neural_2_z)

        activation_record: np.array = np.array([fixed_bias_1, fixed_bias_1, x1, x1, x2, x2, fixed_bias_1, neural_1_activation, neural_2_activation])
        gradient_vector = grdmse(cost, yhat, activation_record)

        weight_vector[0] -= lr * fixed_bias_1 * gradient_vector[0]
        weight_vector[1] -= lr * fixed_bias_1 * gradient_vector[1]
        weight_vector[2] -= lr * x1 * gradient_vector[2]
        weight_vector[3] -= lr * x1 * gradient_vector[3]
        weight_vector[4] -= lr * x2 * gradient_vector[4]
        weight_vector[5] -= lr * x2 * gradient_vector[5]
        weight_vector[6] -= lr * fixed_bias_1 * gradient_vector[6]
        weight_vector[7] -= lr * activation_record[7] * gradient_vector[7]
        weight_vector[8] -= lr * activation_record[8] * gradient_vector[8]



