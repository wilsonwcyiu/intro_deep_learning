

# from __future__ import print_function
from datetime import datetime
from tensorflow.keras.datasets import mnist

import pandas as pd

# from asm_2.hyper_para_mlp_dict import hyper_para_mlp_dict


if __name__ == '__main__':

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(type(x_train))
    print(x_train.shape)