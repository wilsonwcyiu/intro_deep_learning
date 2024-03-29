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


def sigmoid_derivative(activation: np.array):
    return float(sigmoid(activation) * (1 - sigmoid(activation)))


def mse(weights_1: list, weight2: list):
    n = len(weights_1)
    sum: float = 0
    for idx in range(0, n):
        sum += (weights_1[idx] - weight2[idx])**2

    mse: float = sum/n

    return mse



def grdmse(loss_list: list, yhat_list: float, activation_record_list: list):
    # dcost_dpred: float = cost

    weight_vector_grd: list = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for idx in range(0, len(yhat_list)):
        dcost_dpred = float(loss_list[idx]);                                 #print(type(dcost_dpred))

        # output layer to hidden layer grd
        sig_der = sigmoid_derivative(yhat_list[idx]);                           #print("sig_der", type(sig_der))
        weight_vector_grd[6] += dcost_dpred * sig_der;                          #print("weight_vector_grd[6]", type(weight_vector_grd[6]))
        weight_vector_grd[7] += dcost_dpred * sigmoid_derivative(yhat_list[idx])
        weight_vector_grd[8] += dcost_dpred * sigmoid_derivative(yhat_list[idx])

        # hidden layer to input layer grd
        activation_record = activation_record_list[idx]
        weight_vector_grd[0] += weight_vector_grd[7] * sigmoid_derivative(activation_record[7])
        weight_vector_grd[1] += weight_vector_grd[8] * sigmoid_derivative(activation_record[8])

        weight_vector_grd[2] += weight_vector_grd[7] * sigmoid_derivative(activation_record[7])
        weight_vector_grd[3] += weight_vector_grd[8] * sigmoid_derivative(activation_record[8])

        weight_vector_grd[4] += weight_vector_grd[7] * sigmoid_derivative(activation_record[7])
        weight_vector_grd[5] += weight_vector_grd[8] * sigmoid_derivative(activation_record[8])

    return weight_vector_grd

#
# def grdmse(cost: float, yhat_list: float, activation_record_list: list):
#     dcost_dpred: float = cost
#
#     weight_vector_grd: list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#     for idx in range(0, len(yhat_list)):
#         # output layer to hidden layer grd
#         weight_vector_grd[6] += dcost_dpred * sigmoid_derivative(yhat_list[idx])
#         weight_vector_grd[7] += dcost_dpred * sigmoid_derivative(yhat_list[idx])
#         weight_vector_grd[8] += dcost_dpred * sigmoid_derivative(yhat_list[idx])
#
#         # hidden layer to input layer grd
#         activation_record = activation_record_list[idx]
#         weight_vector_grd[0] += weight_vector_grd[7] * sigmoid_derivative(activation_record[7])
#         weight_vector_grd[1] += weight_vector_grd[8] * sigmoid_derivative(activation_record[8])
#
#         weight_vector_grd[2] += weight_vector_grd[7] * sigmoid_derivative(activation_record[7])
#         weight_vector_grd[3] += weight_vector_grd[8] * sigmoid_derivative(activation_record[8])
#
#         weight_vector_grd[4] += weight_vector_grd[7] * sigmoid_derivative(activation_record[7])
#         weight_vector_grd[5] += weight_vector_grd[8] * sigmoid_derivative(activation_record[8])
#
#     return weight_vector_grd



def xor_net(x1_list, x2_list, weight_vector: list):
    activation_record_list: list = []
    yhat_list: list = []
    for idx in range(0, len(x1_list)):
        x1 = x1_list[idx]
        x2 = x2_list[idx]

        # feedforward
        neural_1_z: float = fixed_bias_1 * weight_vector[0] + x1 * weight_vector[2] + x2 * weight_vector[4]
        neural_1_activation: float = sigmoid(neural_1_z)

        neural_2_z: float = fixed_bias_1 * weight_vector[1] + x1 * weight_vector[3] + x2 * weight_vector[5]
        neural_2_activation: float = sigmoid(neural_2_z)

        output_neural_3_z: float = fixed_bias_1 * weight_vector[6] + neural_1_activation * weight_vector[7] + neural_2_activation * weight_vector[8]
        output_activation: float = sigmoid(output_neural_3_z)

        yhat_list.append(output_activation)

        activation_record: list = [fixed_bias_1, fixed_bias_1, x1, x1, x2, x2, fixed_bias_1, neural_1_activation, neural_2_activation]
        activation_record_list.append(activation_record)

    return yhat_list, activation_record_list





if __name__ == '__main__':
    misclassification_count: int = 0
    misclassification_count_list: int = []
    cost_record_list: list = []


    x1: int = [0, 0, 1, 1]
    x2: int = [0, 1, 0, 1]
    all_y_label_list: list = [0, 1, 1, 0]

    fixed_bias_1: int = 1

    # np.random.seed(42)
    weight_vector: list = np.random.rand(9, 1)

    lr = 0.5
    idx = 0
    for epoch in range(200000):
        yhat_list, activation_record_list = xor_net(x1, x2, weight_vector)

        loss_data_list = []
        loss_list: float = []
        for loss_idx in range(0, len(yhat_list)):
            yhat = yhat_list[loss_idx][0]
            loss: float = yhat - all_y_label_list[loss_idx];
            # loss = loss[0]
            loss_data_list.append([str(x1[loss_idx]), str(x2[loss_idx]), str(yhat_list[loss_idx]), str(all_y_label_list[loss_idx])])
            loss_list.append(loss)

        cost = np.sum(loss_list)/ len(loss_list);               #print(cost)
        mse_value = mse(yhat_list, all_y_label_list)

        gradient_desc_vector = grdmse(loss_list, yhat_list, activation_record_list)

        if epoch % 1000 == 0:
            print("\ngradient_desc_vector", gradient_desc_vector)

        for activation_record in activation_record_list:
            weight_vector[0] -= lr * fixed_bias_1         * gradient_desc_vector[0]
            weight_vector[1] -= lr * fixed_bias_1         * gradient_desc_vector[1]
            weight_vector[2] -= lr * activation_record[2] * gradient_desc_vector[2]
            weight_vector[3] -= lr * activation_record[3] * gradient_desc_vector[3]
            weight_vector[4] -= lr * activation_record[4] * gradient_desc_vector[4]
            weight_vector[5] -= lr * activation_record[5] * gradient_desc_vector[5]
            weight_vector[6] -= lr * fixed_bias_1         * gradient_desc_vector[6]
            weight_vector[7] -= lr * activation_record[7] * gradient_desc_vector[7]
            weight_vector[8] -= lr * activation_record[8] * gradient_desc_vector[8]





        yhat_list = list(map(lambda x: 1 if x > 0.5 else 0, yhat_list))
        for idx in range(0, len(yhat_list)):
            if yhat_list[idx] != all_y_label_list[idx]:
                misclassification_count += 1

        # misclassification_count_list.append(misclassification_count)
        # cost_record_list.append(cost)

        # plot_id = 1
        # plot_title = "title"
        # x_label = "Iterations"
        # y_label = "Cost and misclassification count"
        # label_data_list_dict: dict = {"Cost": cost_record_list, "Misclassification": misclassification_count_list}
        if epoch % 1000 == 0:
            print("cost: " + str(cost) + "\t MSE:" + str(mse_value) + "\t Misclassify/Total: " + str(misclassification_count)+ "/" +str(epoch*4) + ". " + str(loss_data_list))
            # plt = plot_multi_list(plot_id, plot_title, x_label, y_label, label_data_list_dict)
            # plt.show()
