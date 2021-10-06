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

    is_show_log: bool = True

    dir_path: str = "D:/Wilson/PycharmProjects/intro_deep_learning/asm_1/data/"
    train_in_file_path: str = dir_path + "train_in.csv"
    train_out_file_path: str = dir_path + "train_out.csv"
    test_in_file_path: str = dir_path + "test_in.csv"
    test_out_file_path: str = dir_path + "test_out.csv"

    train_in_data_list = read_txt_file(train_in_file_path);             print("data row: ", len(train_in_data_list))
    train_out_data_list = read_txt_file(train_out_file_path);           print("data row: ", len(train_out_data_list))
    test_in_data_list = read_txt_file(test_in_file_path);               print("data row: ", len(test_in_file_path))
    test_out_data_list = read_txt_file(test_out_file_path);             print("data row: ", len(test_out_file_path))


    # For each digit d; d = 0; 1; :::; 9; let us consider a cloud of points in 256 dimensional space, Cd, which
    # consists of all training images (vectors) that represent d. For each cloud Cd we can calculate its center,
    # cd, which is just a 256-dimensional vector of means over all coordinates of vectors that belong to Cd.

    # Once we have these centers, we can easily classify new images: by calculating the distance from the
    # vector that represents this image to each of the 10 centers, the closest center defines the label of the
    # image. Next, calculate the distances between the centers of the 10 clouds, distij = dist(ci; cj ), for
    #     i; j = 0; 1; :::9.

    #     Given all these distances, try to say something about the expected accuracy of your
    # classifier. What pairs of digits seem to be most difficult to separate?





    # vector mean of each vector for each digit
    preset_digit_list: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    digit_data_dict: dict = {} # key: digit. value: DigitData
    for result_digit in preset_digit_list:
        digit_data_dict[result_digit] = DigitData(result_digit)

    for i in range(len(train_in_data_list)):
        train_in_data_str = train_in_data_list[i];                      #print(type(train_in_data_str));  print(train_in_data_str);
        train_out_data_str = train_out_data_list[i];                    #print(type(train_out_data_str));  print(train_out_data_str);

        input_data_str_list: list = train_in_data_str.split(",");                       #print(type(input_data_list));  print(input_data_list);
        input_data_list: list = convert_str_list_to_float_list(input_data_str_list);    #print(type(input_data_list));  print(input_data_list);

        output_result: list = float(train_out_data_list[i])

        digit_data_dict[output_result].input_data_list_list.append(input_data_list)


    for output_digit, digitData in digit_data_dict.items():                                     #print("output_digit", digitData.output_digit, "len(digitData.input_data_list_list)", len(digitData.input_data_list_list))

        input_np_matrix: list = np.array(digitData.input_data_list_list);                                                   #print(input_np_matrix.shape)
        digitData.vector_of_means_list: list = calculate_mean_for_matrix(input_np_matrix, direction="col");     #print("digitData.vector_of_means_list.shape", digitData.vector_of_means_list.shape)




    # calculate distance matrix
    all_digit_distance_matrix_summary: list = []
    for i in range(10):
        each_digit_distance_matrix_list: list = []
        for j in range(10):
            vector_of_means_matrix_a: list = digit_data_dict[i].vector_of_means_list
            vector_of_means_matrix_b: list = digit_data_dict[j].vector_of_means_list

            distance = vector_of_means_matrix_b - vector_of_means_matrix_a
            distance = np.absolute(distance)
            distance = numpy.around(distance, decimals=5)
            sum = calculate_sum_for_matrix(distance, direction="all")

            each_digit_distance_matrix_list.append(sum)

        all_digit_distance_matrix_summary.append(each_digit_distance_matrix_list)



    all_digit_distance_matrix_summary = numpy.around(all_digit_distance_matrix_summary, decimals=1)
    for all_digit_distance_matrix_list in all_digit_distance_matrix_summary:
        print(all_digit_distance_matrix_list)


    # print(distance)




