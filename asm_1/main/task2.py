import pprint

import numpy as np

from matplotlib import pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


if __name__ == '__main__':



    feature_set: np.array = np.array([
        #[0,1,0],[0,0,1],[1,0,0],[1,1,0],
        [1,1,1]])


    y_label: np.array = np.array([[
        # 1, 0, 0, 1,
        1]])
    # y_label: np.array = y_label.reshape(5, 1)
    y_label: np.array = y_label.reshape(1, 1)

    np.random.seed(42)
    weights: np.array = np.random.rand(3,1)
    print(type(weights))
    print(weights)
    print(weights.shape)


    bias: np.array = np.random.rand(1)
    print(type(bias))
    print(bias)
    print(bias.shape)


    lr = 0.05


    for _ in range(200000):


        # feedforward step1
        z: np.array = np.dot(feature_set, weights) + bias;
        # print("z", z)

        #feedforward step2
        yhat_activation: np.array = sigmoid(z);         print("yhat_activation.shape", yhat_activation.shape)
        # print("activation", yhat_activation)

        # backpropagation step 1
        error: np.array = yhat_activation - y_label
        # print("error", error)
        print(error.sum())

        # backpropagation step 2
        derivative_cost_over_derivative_dpred: np.array = error
        derivative_pred_over_derivative_z: np.array = sigmoid_derivative(yhat_activation)

        z_delta: np.array = derivative_cost_over_derivative_dpred * derivative_pred_over_derivative_z
        # print("z_delta: ", z_delta)

        inputs = feature_set.T
        weights -= lr * np.dot(inputs, z_delta)

        for num in z_delta:
            bias -= lr * num
