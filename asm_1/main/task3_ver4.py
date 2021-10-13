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











if __name__ == '__main__':
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
    hidden_layer_weight: np.array = np.random.rand(3, 2);                #print(type(hidden_layer_W), hidden_layer_W, hidden_layer_W.shape)
    hidden_layer_bias: np.array = np.random.rand(1, 2);             #print(type(hidden_layer_bias), hidden_layer_bias, hidden_layer_bias.shape)

    output_layer_weight: np.array = np.random.rand(3, 1);                #print(type(output_layer_W), output_layer_W, output_layer_W.shape)
    output_layer_bias: np.array = np.random.rand(1, 3);             #print(type(output_layer_bias), output_layer_bias, output_layer_bias.shape)


    lr = 0.0005

    for epoch in range(200000):

        # feedforward fromm input layer to hidden layer
        layer_1_z: np.array = np.dot(all_train_set, hidden_layer_weight);           #print("z1 shape", z1.shape)
        layer_1_z += hidden_layer_bias
        layer_1_activation: np.array = sigmoid(layer_1_z)


        # feedforward from hidden layer to output layer
        # add 1 bias input as required from assignment
        loc: int = 0
        layer_1_activation = np.insert(layer_1_activation, loc, fixed_bias_1, axis=1);              #print(">>a1", a1)

        layer_2_z: np.array = np.dot(layer_1_activation, output_layer_weight);           #print("z2", z2)
        layer_2_activation_yhat: np.array = sigmoid(layer_2_z);                        #print("a2_yhat_activation", a2)



        # calculate lost and cost
        lost_data_array: np.array = layer_2_activation_yhat - all_y_label;
        cost: float = lost_data_array.sum();                                            print("Cost: ", cost)


        # backpropagation from output Y to output layer
        output_layer_dcost_dpred: np.array = lost_data_array;                                          #print("dcost_dpred.shape", dcost_dpred.shape)
        output_layer_dpred_dz: np.array = sigmoid_derivative(layer_2_activation_yhat);               #print("dpred_dz.shape", dpred_dz.shape)

        output_layer_activation_delta: np.array = output_layer_dcost_dpred * output_layer_dpred_dz;                             #print("z_delta.shape", z_delta.shape)

        output_layer_weight -= lr * np.dot(layer_1_activation.T, output_layer_activation_delta);       #print("output_layer_W.shape", output_layer_W.shape)

        for activation_delta in output_layer_activation_delta:
            output_layer_weight -= lr * activation_delta


        # backpropagation from output layer to hidden layer
        hidden_layer_error: np.array = layer_1_activation - output_layer_weight.T;       #print("hl_error.shape", hl_error.shape)
        hidden_layer_dcost_dpred: np.array = lost_data_array;
        hidden_layer_dpred_dz: np.array = sigmoid_derivative(layer_1_activation);     #print("hl_dpred_dz.shape", hl_dpred_dz.shape)

        hidden_layer_activation_delta: np.array = hidden_layer_dcost_dpred * hidden_layer_dpred_dz;                             #print("hl_z_delta.shape", hl_z_delta.shape)
        loc: int = 0; axis = 1

        # remove first bias that is manually added in previous process
        hidden_layer_activation_delta = np.delete(hidden_layer_activation_delta, loc, axis);

        hidden_layer_weight -= lr * np.dot(all_train_set.T, hidden_layer_activation_delta);       #print("hidden_layer_W.shape", hidden_layer_W.shape)

        for activation_delta in hidden_layer_activation_delta:
            hidden_layer_bias -= lr * activation_delta











