import pprint

import numpy as np

from matplotlib import pyplot as plt


class DigitData():

    def __init__(self, output_digit: int):
        self.output_digit: int = output_digit
        self.input_data_list_list: list = []        # 2 dimensional matrix. each row = 1 input data. each column = pixel value of same coordinate
        self.vector_of_means_list: list = None



def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


def read_txt_file(abs_file_path: str):
    file = open(abs_file_path, "r")
    data_list: list = file.readlines()
    file.close()

    return data_list


def convert_str_list_to_float_list(str_list: list):
    float_list = [float(i) for i in str_list]

    return float_list


if __name__ == '__main__':


    dir_path: str = "D:/Wilson/PycharmProjects/intro_deep_learning/asm_1/data/"
    train_in_file_path: str = dir_path + "train_in.csv"
    train_out_file_path: str = dir_path + "train_out.csv"
    test_in_file_path: str = dir_path + "test_in.csv"
    test_out_file_path: str = dir_path + "test_out.csv"

    train_in_data_list = read_txt_file(train_in_file_path);             print("data row: ", len(train_in_data_list))
    train_out_data_list = read_txt_file(train_out_file_path);           print("data row: ", len(train_out_data_list))
    test_in_data_list = read_txt_file(test_in_file_path);               print("data row: ", len(test_in_file_path))
    test_out_data_list = read_txt_file(test_out_file_path);             print("data row: ", len(test_out_file_path))


    print("pre process data")
    # preset_digit_list: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # digit_data_dict: dict = {} # key: digit. value: DigitData
    # for result_digit in preset_digit_list:
    #     digit_data_dict[result_digit] = DigitData(result_digit)




    feature_list: list = []
    y_label_list: list = []
    for i in range(len(train_in_data_list)):
        train_in_data_str = train_in_data_list[i];                      #print(type(train_in_data_str));  print(train_in_data_str);
        train_out_data_str = train_out_data_list[i];                    #print(type(train_out_data_str));  print(train_out_data_str);

        input_data_str_list: list = train_in_data_str.split(",");                       #print(type(input_data_list));  print(input_data_list);
        input_data_list: list = convert_str_list_to_float_list(input_data_str_list);    #print(type(input_data_list));  print(input_data_list);

        output_result: list = int(train_out_data_list[i])

        feature_list.append(input_data_list)
        y_label_list.append(output_result)



    feature_set: np.array = np.array(feature_list)
    y_label: np.array = np.array([y_label_list])

    num_of_y_label: int = y_label.shape[1]
    y_label: np.array = y_label.reshape(num_of_y_label, 1)

    np.random.seed(42)
    num_of_case: int = feature_set.shape[0]
    input_data_size: int = feature_set.shape[1]
    weights: np.array = np.random.rand(input_data_size, num_of_case)
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
