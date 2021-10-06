

def read_txt_file(abs_file_path: str):
    file = open(abs_file_path, "r")
    data_list: list = file.readlines()
    file.close()

    return data_list


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


    digit_mean_dict: dict = {}
    for i in range(len(train_in_data_list)):
        train_in_data = train_in_data_list[i]
        train_out_data = train_out_data_list[i]

        vector_mean: float = 0

        if train_out_data not in digit_mean_dict.keys():
            digit_mean_dict[train_out_data] = []

        digit_mean_dict[train_out_data].append(vector_mean)




    # mean of each vector for each digit

    # calculate distance
