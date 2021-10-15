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



    # all_train_set: np.array = np.array([
    #     [1,1,0],
    #     [1,0,1],
    #     [1,0,0],
    #     [1,1,0],
    #     [1,1,1]
    # ]);                             #pprint.pprint(all_train_set);pprint.pprint(all_train_set.shape)
    # all_y_label: np.array = np.array([[
    #     1, 1, 0, 1,
    #     0]])
    #

    # all_train_set: np.array = np.array([
    #     [0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]
    # ]);                             #pprint.pprint(all_train_set);pprint.pprint(all_train_set.shape)
    # all_y_label: np.array = np.array([[
    #     1, 0, 0, 1, 1
    # ]])
    # all_y_label: np.array = all_y_label.reshape(5, 1)


    fixed_bias_1 = 1
    all_train_set: np.array = np.array([
        [fixed_bias_1,0,0],[fixed_bias_1,0,1],[fixed_bias_1,1,0],[fixed_bias_1,1,1]
    ]);
    all_y_label: np.array = np.array([[
        0, 1, 1, 0
    ]])

    all_y_label: np.array = all_y_label.reshape(4, 1)

    np.random.seed(42)
    hidden_layer_W: np.array = np.random.rand(3, 2);                #print(type(hidden_layer_W), hidden_layer_W, hidden_layer_W.shape)
    # hidden_layer_bias: np.array = np.random.rand(1, 2);             #print(type(hidden_layer_bias), hidden_layer_bias, hidden_layer_bias.shape)

    output_layer_W: np.array = np.random.rand(3, 1);                #print(type(output_layer_W), output_layer_W, output_layer_W.shape)
    # output_layer_bias: np.array = np.random.rand(1, 3);             #print(type(output_layer_bias), output_layer_bias, output_layer_bias.shape)


    lr = 0.0005

    idx = 0


    for epoch in range(200000):
        if idx == len(all_train_set) - 1: idx = 0
        else:                             idx += 1

        # random_idx: int = random.randrange(0, len(all_train_set))
        # inputs: np.array = feature_set
        # #print(inputs)

        # one_train_set: np.array = all_train_set[random_idx]
        # one_result =
        # feedforward step1

        # train_set = np.array([all_train_set[idx]]);                  # print("train_set shape", train_set.shape)
        # y_label = np.array([all_y_label[idx]]);                       #print("y_label shape", y_label.shape)
        train_set = all_train_set;                  # print("train_set shape", train_set.shape)
        y_label = all_y_label;                       #print("y_label shape", y_label.shape)

        z1: np.array = np.dot(train_set, hidden_layer_W);           #print("z1 shape", z1.shape)
        # z1 += hidden_layer_bias
        a1: np.array = sigmoid(z1)

        loc: int = 0
        static_bias: int = 1

        a1 = np.insert(a1, loc, static_bias, axis=1);           #print(">>a1", a1)
        z2: np.array = np.dot(a1, output_layer_W);           #print("z2", z2)
        a2: np.array =  sigmoid(z2);                        #print("a2", a2)



        #feedforward step2
        yhat_activation: np.array = sigmoid(z2)
        # #print("activation", yhat_activation)


        # backpropagation step 1
        error: np.array = yhat_activation - y_label;
        mse = 0
        for err in error:
            mse += err**2

        mse = mse / len(error);                                             print("mse", mse, "error.sum()", error.sum())

        dcost_dpred: np.array = error;                                          #print("dcost_dpred.shape", dcost_dpred.shape)
        dpred_dz: np.array = sigmoid_derivative(yhat_activation);               #print("dpred_dz.shape", dpred_dz.shape)

        z_delta: np.array = dcost_dpred * dpred_dz;                             #print("z_delta.shape", z_delta.shape)

        # inputs = all_train_set;       #print("inputs.shape", inputs.shape)
        output_layer_W -= lr * np.dot(a1.T, z_delta);       #print("output_layer_W.shape", output_layer_W.shape)

        # for num in z_delta:
        #     output_layer_W -= lr * num
        #print("output_layer_W", output_layer_W, "output_layer_W.shape", output_layer_W.shape)


        hl_error: np.array = a1 - output_layer_W.T;       #print("hl_error.shape", hl_error.shape)
        hl_dcost_dpred: np.array = error;
        hl_dpred_dz: np.array = sigmoid_derivative(a1);     #print("hl_dpred_dz.shape", hl_dpred_dz.shape)

        hl_z_delta: np.array = hl_dcost_dpred * hl_dpred_dz;                             #print("hl_z_delta.shape", hl_z_delta.shape)
        loc: int = 0; axis = 1

        # remove first bias
        hl_z_delta = np.delete(hl_z_delta, loc, axis);

        hidden_layer_W -= lr * np.dot(train_set.T, hl_z_delta);       #print("hidden_layer_W.shape", hidden_layer_W.shape)

        # for num in hl_z_delta:
        #     hidden_layer_bias -= lr * num











