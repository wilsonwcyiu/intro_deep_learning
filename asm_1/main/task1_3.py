import numpy

import numpy as np
from pandas import DataFrame


def read_txt_file(abs_file_path: str):
    file = open(abs_file_path, "r")
    data_list: list = file.readlines()
    file.close()

    return data_list


def convert_str_list_to_float_list(str_list: list):
    float_list = [float(i) for i in str_list]

    return float_list


def calculate_mean_for_matrix(input_matrix: list, direction: str):
    if direction == "row": direction_idx = 1
    elif direction == "col": direction_idx = 0
    elif direction == "all": direction_idx = None
    else: raise Exception(direction)

    result_mean = input_matrix.mean(axis=direction_idx)

    return result_mean


def calculate_sum_for_matrix(input_matrix: list, direction: str):
    if direction == "row": direction_idx = 1
    elif direction == "col": direction_idx = 0
    elif direction == "all": direction_idx = None
    else: raise Exception(direction)

    result_sum = input_matrix.sum(axis=direction_idx)

    return result_sum


class DigitData():

    def __init__(self, output_digit: int):
        self.output_digit: int = output_digit
        self.input_data_list_list: list = []        # 2 dimensional matrix. each row = 1 input data. each column = pixel value of same coordinate
        self.vector_of_means_list: list = None


if __name__ == '__main__':
    dir_path: str = "D:/Wilson/PycharmProjects/intro_deep_learning/asm_1/data/"
    train_in_file_path: str = dir_path + "train_in.csv"
    train_out_file_path: str = dir_path + "train_out.csv"
    test_in_file_path: str = dir_path + "test_in.csv"
    test_out_file_path: str = dir_path + "test_out.csv"

    train_in_data_list = read_txt_file(train_in_file_path);           print("data row: ", len(train_in_data_list))
    train_out_data_list = read_txt_file(train_out_file_path);           print("data row: ", len(train_out_data_list))
    test_in_data_list = read_txt_file(test_in_file_path);           print("data row: ", len(test_in_file_path))
    test_out_data_list = read_txt_file(test_out_file_path);           print("data row: ", len(test_out_file_path))



    # Implement the simplest distance-based classifier that is described in part 1.
    # Apply your classifier to all points from the training set and calculate the percentage of correctly classified digits.
    # Do the same with the test set, using the centers that were calculated from the training set.



    # vector mean of each vector for each digit
    preset_digit_list: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    digit_data_dict: dict = {} # key: digit. value: DigitData
    for result_digit in preset_digit_list:
        digit_data_dict[result_digit] = DigitData(result_digit)

    for i in range(len(train_in_data_list)):
        test_in_data_str = train_in_data_list[i];                      #print(type(train_in_data_str));  print(train_in_data_str);
        test_out_data_str = train_out_data_list[i];                    #print(type(train_out_data_str));  print(train_out_data_str);

        input_data_str_list: list = test_in_data_str.split(",");                       #print(type(input_data_list));  print(input_data_list);
        input_data_list: list = convert_str_list_to_float_list(input_data_str_list);    #print(type(input_data_list));  print(input_data_list);

        output_result: list = float(train_out_data_list[i])

        digit_data_dict[output_result].input_data_list_list.append(input_data_list)


    for output_digit, digitData in digit_data_dict.items():                                     #print("output_digit", digitData.output_digit, "len(digitData.input_data_list_list)", len(digitData.input_data_list_list))

        input_np_matrix: list = np.array(digitData.input_data_list_list);                                                   #print(input_np_matrix.shape)
        digitData.vector_of_means_list: list = calculate_mean_for_matrix(input_np_matrix, direction="col");     #print("digitData.vector_of_means_list.shape", digitData.vector_of_means_list.shape)






    # classify from train set
    for i in range(len(train_in_data_list)):
        test_in_data_str = train_in_data_list[i];                      #print(type(train_in_data_str));  print(train_in_data_str);
        test_out_data_str = train_out_data_list[i];                    #print(type(train_out_data_str));  print(train_out_data_str);

        input_data_str_list: list = test_in_data_str.split(",");                       #print(type(input_data_list));  print(input_data_list);
        input_data_list: list = convert_str_list_to_float_list(input_data_str_list);    #print(type(input_data_list));  print(input_data_list);

        output_result: list = int(train_out_data_list[i])

        classifier_result: int = None
        min_distance_record: float = 999999

        for digitData in digit_data_dict.values():
            distance: list = input_data_list - digitData.vector_of_means_list
            distance = np.absolute(distance)
            distance = numpy.around(distance, decimals=5)
            sum = calculate_sum_for_matrix(distance, direction="all")

            if sum <= min_distance_record:
                min_distance_record: float = sum
                classifier_result = digitData.output_digit

        print("result from training set -- classifier_result(y_hat), true_result (y): ", classifier_result, output_result)





    # classify from test set
    for i in range(len(test_in_data_list)):
        test_in_data_str = test_in_data_list[i];                      #print(type(train_in_data_str));  print(train_in_data_str);
        test_out_data_str = test_out_data_list[i];                    #print(type(train_out_data_str));  print(train_out_data_str);

        input_data_str_list: list = test_in_data_str.split(",");                       #print(type(input_data_list));  print(input_data_list);
        input_data_list: list = convert_str_list_to_float_list(input_data_str_list);    #print(type(input_data_list));  print(input_data_list);

        output_result: list = int(train_out_data_list[i])

        classifier_result: int = None
        min_distance_record: float = 999999

        for digitData in digit_data_dict.values():
            distance: list = input_data_list - digitData.vector_of_means_list
            distance = np.absolute(distance)
            distance = numpy.around(distance, decimals=5)
            sum = calculate_sum_for_matrix(distance, direction="all")

            if sum <= min_distance_record:
                min_distance_record: float = sum
                classifier_result = digitData.output_digit

        print("result from testing set -- classifier_result(y_hat), true_result (y): ", classifier_result, output_result)




