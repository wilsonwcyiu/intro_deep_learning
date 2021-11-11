import itertools

# from keras.optimizer_v1 import RMSprop
# import tensorflow
# from tensorflow import keras
# from tensorflow.keras import optimizers
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import RMSprop
# import tensorflow.keras.optimizers
import itertools
import numpy as np
import pandas as pd


if __name__ == '__main__':

    dict_idx: int = 20000
    epoch_list = [10]
    initializer_list = []
    # initializer_list.append("random_normal")
    initializer_list.append("uniform")
    initializer_list.append("tensorflow.keras.initializers.RandomNormal(stddev=0.01)")
    # initializer_list.append(tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))
    # initializer_list.append(tensorflow.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None))

    network_structure_id_list = [1, 2]

    activation_list = ["relu", "sigmoid"]

    regularization_L1_list = [0, 0.1]
    regularization_L2_list = [0, 0.1, 0.001]
    drop_out_rate_list = [0, 0.2]

    loss_func_list = ["categorical_crossentropy", "mse"]
    optimizers_list = ["optimizers.Adam(learning_rate=0.01)", "sgd"]



    all_options_list = []
    all_options_list.append(epoch_list)
    all_options_list.append(initializer_list)
    all_options_list.append(network_structure_id_list)
    all_options_list.append(activation_list)
    all_options_list.append(regularization_L1_list)
    all_options_list.append(regularization_L2_list)
    all_options_list.append(drop_out_rate_list)
    all_options_list.append(loss_func_list)
    all_options_list.append(optimizers_list)

    combination_tuple_list = list(itertools.product(*all_options_list))

    print(len(combination_tuple_list))



    for combination_tuple in combination_tuple_list:
        epoch = combination_tuple[0]
        initializer = combination_tuple[1]
        network_structure_id = combination_tuple[2]
        activation = "\"" + combination_tuple[3] + "\""
        regularization_L1 = combination_tuple[4]
        regularization_L2 = combination_tuple[5]
        drop_out_rate = combination_tuple[6]
        loss_func = "\"" + combination_tuple[7] + "\""
        optimizer = combination_tuple[8]

        if "(" not in initializer: initializer = "\"" + initializer + "\""
        if "(" not in optimizer: optimizer = "\"" + optimizer + "\""


        template: str = f"{dict_idx}: HyperParaCnn(epoch={epoch}," \
                                                    f"initializer={initializer}, " \
                                                    f"network_structure_id={network_structure_id}, " \
                                                    f"activation={activation}, " \
                                                    f"regularization_L1={regularization_L1}, " \
                                                    f"regularization_L2={regularization_L2}, " \
                                                    f"drop_out_rate={drop_out_rate}, " \
                                                    f"loss_func={loss_func}, " \
                                                    f"optimizers={optimizer}), "

        dict_idx += 1

        print(template)

    print(f"next start dict_idx: {dict_idx}")