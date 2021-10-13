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


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


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
    hidden_layer_W: np.array = np.random.rand(3, 2);                #print(type(hidden_layer_W), hidden_layer_W, hidden_layer_W.shape)
    hidden_layer_bias: np.array = np.random.rand(1, 2);             #print(type(hidden_layer_bias), hidden_layer_bias, hidden_layer_bias.shape)

    output_layer_W: np.array = np.random.rand(3, 1);                #print(type(output_layer_W), output_layer_W, output_layer_W.shape)
    output_layer_bias: np.array = np.random.rand(1, 3);             #print(type(output_layer_bias), output_layer_bias, output_layer_bias.shape)


    lr = 0.0005

    for epoch in range(200000):


        # feedforward fromm input layer to hidden layer
        z1: np.array = np.dot(all_train_set, hidden_layer_W);           #print("z1 shape", z1.shape)
        z1 += hidden_layer_bias
        a1: np.array = sigmoid(z1)


        # feedforward from hidden layer to output layer
        # add 1 bias input as required from assignment
        loc: int = 0
        a1 = np.insert(a1, loc, fixed_bias_1, axis=1);              #print(">>a1", a1)

        z2: np.array = np.dot(a1, output_layer_W);           #print("z2", z2)
        a2_yhat_activation: np.array = sigmoid(z2);                        #print("a2_yhat_activation", a2)



        # calculate cost
        error: np.array = a2_yhat_activation - all_y_label;                         print("error.sum()", error.sum())



        # backpropagation from output Y to output layer
        dcost_dpred: np.array = error;                                          #print("dcost_dpred.shape", dcost_dpred.shape)
        dpred_dz: np.array = sigmoid_derivative(a2_yhat_activation);               #print("dpred_dz.shape", dpred_dz.shape)

        z_delta: np.array = dcost_dpred * dpred_dz;                             #print("z_delta.shape", z_delta.shape)

        output_layer_W -= lr * np.dot(a1.T, z_delta);       #print("output_layer_W.shape", output_layer_W.shape)

        for num in z_delta:
            output_layer_W -= lr * num


        # backpropagation from output layer to hidden layer
        hl_error: np.array = a1 - output_layer_W.T;       #print("hl_error.shape", hl_error.shape)
        hl_dcost_dpred: np.array = error;
        hl_dpred_dz: np.array = sigmoid_derivative(a1);     #print("hl_dpred_dz.shape", hl_dpred_dz.shape)

        hl_z_delta: np.array = hl_dcost_dpred * hl_dpred_dz;                             #print("hl_z_delta.shape", hl_z_delta.shape)
        loc: int = 0; axis = 1

        # remove first bias as it is manually added in previous process
        hl_z_delta = np.delete(hl_z_delta, loc, axis);

        hidden_layer_W -= lr * np.dot(all_train_set.T, hl_z_delta);       #print("hidden_layer_W.shape", hidden_layer_W.shape)

        for num in hl_z_delta:
            hidden_layer_bias -= lr * num











