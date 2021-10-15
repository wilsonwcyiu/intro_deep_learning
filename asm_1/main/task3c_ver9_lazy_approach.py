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
    return (1/(1 + np.exp(-z)))


def sigmoid_derivative(activation: np.array):
    return sigmoid(activation) * (1 - sigmoid(activation))




def mse(weights_1: list, weight2: list):
    n = len(weights_1)
    sum: float = 0
    for idx in range(0, n):
        sum += (weights_1[idx] - weight2[idx])**2

    mse: float = sum/n

    return mse



from matplotlib import pyplot
def plot_multi_list(plot_id: int, plot_title: str, x_label: str, y_label: str, label_data_list_dict: dict):
    pyplot.figure(plot_id)
    pyplot.clf()

    pyplot.title(plot_title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    for key, item in label_data_list_dict.items():
        pyplot.plot(item, label=key)

    pyplot.legend()

    return pyplot




if __name__ == '__main__':
    mse_data_dict: dict = {}
    misclassification_count_dict = {}

    fixed_bias_1: int = 1

    all_train_set: np.array = np.array([
        [fixed_bias_1,0,0],[fixed_bias_1,0,1],[fixed_bias_1,1,0],[fixed_bias_1,1,1]
    ]);
    all_y_label: np.array = np.array([[
        0, 1, 1, 0
    ]])
    all_y_label: np.array = all_y_label.reshape(4, 1)


    # initial weights
    np.random.seed(42)

    misclassification_count: int = 0
    misclassification_count_list: int = []

    data_list: list = []
    for epoch in range(100000):

        low: float = -1
        high: float = 1
        hidden_layer_weight = np.random.rand(3, 2) * (high-low) + low
        output_layer_weight = np.random.rand(3, 1) * (high-low) + low

        # feedforward from input layer to hidden layer
        layer_1_z: np.array = np.dot(all_train_set, hidden_layer_weight);           #print("z1 shape", z1.shape)
        layer_1_activation: np.array = sigmoid(layer_1_z)


        # feedforward from hidden layer to output layer
        # add 1 bias input as required from assignment
        loc: int = 0
        layer_1_activation = np.insert(layer_1_activation, loc, fixed_bias_1, axis=1);              #print(">>a1", a1)

        layer_2_z: np.array = np.dot(layer_1_activation, output_layer_weight);           #print("z2", z2)
        layer_2_activation_yhat: np.array = sigmoid(layer_2_z);                        #print("a2_yhat_activation", a2)


        # calculate lost and cost
        lost_data_array: np.array = layer_2_activation_yhat - all_y_label;
        cost: float = lost_data_array.sum();
        mse_value = mse(layer_2_activation_yhat, all_y_label);                           #print("Cost: ", cost, "mse", mse_value)

        data_list.append(mse_value)

        # count misclassification
        for idx in range(len(layer_2_activation_yhat)):
            yhat_binary = 1 if lost_data_array[idx] > 0.5 else 0
            if yhat_binary != all_y_label[idx]:
                misclassification_count += 1

        misclassification_count_list.append(misclassification_count)

    mse_data_dict["lazy approach"] = data_list
    misclassification_count_dict["lazy approach"] = misclassification_count_list

    plot_id = 1
    plot_title = "MSE in lazy approach"
    x_label = "Iterations"
    y_label = "MSE"
    plt = plot_multi_list(plot_id, plot_title, x_label, y_label, mse_data_dict)
    plt.show()


    plot_id = 2
    plot_title = "Misclassification count in lazy approach"
    x_label = "Iterations"
    y_label = "Misclassification count"
    plt = plot_multi_list(plot_id, plot_title, x_label, y_label, misclassification_count_dict)
    plt.show()