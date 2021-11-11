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
    return sigmoid(activation) * (1 - sigmoid(activation))


def mse(weights_1: list, weight2: list):
    n = len(weights_1)
    sum: float = 0
    for idx in range(0, n):
        sum += (weights_1[idx] - weight2[idx])**2

    mse: float = sum/n

    return mse


def mse_from_loss(loss_list):
    n = len(loss_list)
    sum: float = 0
    for idx in range(0, n):
        sum += loss_list[idx]**2

    mse: float = sum/n

    return mse



# def grdmse(loss_list: list, yhat_list: float, activation_record_list: list):
#     # dcost_dpred: float = cost
#
#     weight_vector_grd: list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#     for idx in range(0, len(yhat_list)):
#         dcost_dpred = loss_list[idx]
#
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

#
def grdmse(cost: float, yhat: float, neural_1_activation, neural_2_activation):
    dcost_dpred: float = cost

    weight_vector_grd: list = [0, 0, 0, 0, 0, 0, 0, 0, 0]


    # output layer to hidden layer grd
    weight_vector_grd[6] += dcost_dpred * sigmoid_derivative(yhat)
    weight_vector_grd[7] += dcost_dpred * sigmoid_derivative(yhat)
    weight_vector_grd[8] += dcost_dpred * sigmoid_derivative(yhat)

    # hidden layer to input layer grd
    # activation_record = activation_record_list[idx]
    weight_vector_grd[0] += weight_vector_grd[7] * sigmoid_derivative(neural_1_activation)
    weight_vector_grd[1] += weight_vector_grd[8] * sigmoid_derivative(neural_2_activation)

    weight_vector_grd[2] += weight_vector_grd[7] * sigmoid_derivative(neural_1_activation)
    weight_vector_grd[3] += weight_vector_grd[8] * sigmoid_derivative(neural_2_activation)

    weight_vector_grd[4] += weight_vector_grd[7] * sigmoid_derivative(neural_1_activation)
    weight_vector_grd[5] += weight_vector_grd[8] * sigmoid_derivative(neural_2_activation)

    return weight_vector_grd



def xor_net(x1, x2, weight_vector: list):

    # feedforward
    neural_1_z: float = fixed_bias_1 * weight_vector[0] + x1 * weight_vector[2] + x2 * weight_vector[4]
    neural_1_activation: float = sigmoid(neural_1_z)

    neural_2_z: float = fixed_bias_1 * weight_vector[1] + x1 * weight_vector[3] + x2 * weight_vector[5]
    neural_2_activation: float = sigmoid(neural_2_z)

    output_neural_3_z: float = fixed_bias_1 * weight_vector[6] + neural_1_activation * weight_vector[7] + neural_2_activation * weight_vector[8]
    output_activation: float = sigmoid(output_neural_3_z)


    return output_activation, neural_1_activation, neural_2_activation





if __name__ == '__main__':
    misclassification_count: int = 0
    misclassification_count_list: int = []
    loss_record_list: list = []


    x1_list: int = [0, 0, 1, 1]
    x2_list: int = [0, 1, 0, 1]
    all_y_label_list: list = [0, 1, 1, 0]

    fixed_bias_1: int = 1

    # np.random.seed(42)
    weight_vector: list = np.random.rand(9, 1)

    lr = 0.001
    idx = 0
    for epoch in range(200000):
        # print(epoch)
        x1 = x1_list[idx]
        x2 = x2_list[idx]

        idx += 1
        if idx == 4:
            idx = 0


        yhat, neural_1_activation, neural_2_activation = xor_net(x1_list[idx], x2_list[idx], weight_vector)
        y_label = all_y_label_list[idx]


        loss: float = yhat - y_label;
        loss_record_list.append(loss)

        # cost = np.sum(loss_list)/ len(loss_list);               #print(cost)
        # mse_value = mse(yhat_list, all_y_label_list)

        gradient_desc_vector = grdmse(loss, yhat, neural_1_activation, neural_2_activation)

        # for activation_record in activation_record_list:
        weight_vector[0] -= lr * fixed_bias_1 * gradient_desc_vector[0]
        weight_vector[1] -= lr * fixed_bias_1 * gradient_desc_vector[1]
        weight_vector[2] -= lr * x1 * gradient_desc_vector[2]
        weight_vector[3] -= lr * x1 * gradient_desc_vector[3]
        weight_vector[4] -= lr * x2 * gradient_desc_vector[4]
        weight_vector[5] -= lr * x2 * gradient_desc_vector[5]
        weight_vector[6] -= lr * fixed_bias_1 * gradient_desc_vector[6]
        weight_vector[7] -= lr * neural_1_activation * gradient_desc_vector[7]
        weight_vector[8] -= lr * neural_2_activation * gradient_desc_vector[8]





        # yhat_list = list(map(lambda x: 1 if x > 0.5 else 0, yhat_list))
        # for idx in range(0, len(yhat_list)):
        #     if yhat_list[idx] != all_y_label_list[idx]:
        #         misclassification_count += 1

        # misclassification_count_list.append(misclassification_count)
        # cost_record_list.append(cost)

        # plot_id = 1
        # plot_title = "title"
        # x_label = "Iterations"
        # y_label = "Cost and misclassification count"
        # label_data_list_dict: dict = {"Cost": cost_record_list, "Misclassification": misclassification_count_list}
        if epoch % 1000 == 0:
            mse_value = mse_from_loss(loss_record_list[-40:])
            print(mse_value)



            # print(str(epoch*4) + "," + str(cost) + "," + str(mse_value) + "," + str(misclassification_count))
            # plt = plot_multi_list(plot_id, plot_title, x_label, y_label, label_data_list_dict)
            # plt.show()
