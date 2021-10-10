

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



    # A less naive distance-based approach is the KNN (K-Nearest-Neighbor) classier (you can either implement it yourself or use the one from sklearn package).
    ## By using this method repeat the same procedure as in part 3.

    # Then, for both classiers generate a confusion matrix which should provide a deeper insight into classes that are dicult to separate.
    # A confusion matrix is here a 10-by-10 matrix (cij ), where cij contains the percentage (or count) of digits i that are classied as j.
    ## Which digits are most dicult to classify correctly?

    # Again, for calculating and visualising confusion matrices you may use the sklearn package.
    ## Describe your ndings, compare performance of your classiers on the train and test sets.



    digit_mean_dict: dict = {}
    for i in range(len(train_in_data_list)):
        train_in_data = train_in_data_list[i]
        train_out_data = train_out_data_list[i]

        vector_mean: float = 0

        if train_out_data not in digit_mean_dict.keys():
            digit_mean_dict[train_out_data] = []

        digit_mean_dict[train_out_data].append(vector_mean)



    # KNN

    # confusion matrix
