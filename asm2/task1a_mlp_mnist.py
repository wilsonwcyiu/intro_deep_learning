'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''


# from __future__ import print_function
import tensorflow
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import itertools
import numpy as np
import tensorflow.keras.optimizers

if __name__ == '__main__':

    # fixed para
    batch_size = 128
    num_classes = 10
    epochs = 2


    # variations
    layers_id: int = 1

    epoch_list = []

    initializer_list = [
                        "random_normal",
                        "uniform",
                        tensorflow.keras.initializers.RandomNormal(stddev=0.01),
                        tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                        tensorflow.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
                    ]

    layer_tuple_list = [
                        (512, 512),
                        (512, 256),
                        (256, 128)
                    ]
    activation_list = ["relu", "sigmoid", "tanh"]
    regularization_L1_list = [0, 0.1, 0.001, 0.0001]
    regularization_L2_list = [0, 0.1, 0.001, 0.0001]
    drop_out_rate_list = [0, 0.2, 0.4]

    loss_func_list = ["categorical_crossentropy", "mse", "mae", "log_cosh"]

    optimizers_list = [RMSprop(), optimizers.Adam(learning_rate=0.01), "sgd"]



    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_size = 50000
    x_train = x_train[0:train_size]
    y_train = y_train[0:train_size]

    test_size = 5000
    x_test = x_test[0:test_size]
    y_test = y_test[0:test_size]

    # print(x_train.shape)
    # exit()


    x_train = x_train.reshape(train_size, 784)
    x_train = x_train.astype('float32')
    x_train /= 255
    # print(x_train.shape[0], 'train samples')

    x_test = x_test.reshape(test_size, 784)
    x_test = x_test.astype('float32')
    x_test /= 255
    # print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(784,)))
    # model.add(Dense(512, activation='relu', input_shape=(784,), kernel_regularizer=keras.regularizers.l2(0)))
    initializer = "random_normal"
    kernel_regularizer = keras.regularizers.l1_l2(l1=0, l2=0)
    model.add(Dense(512, activation='relu', input_shape=(784,), kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # model.summary()

    model.compile(loss="mse", #'categorical_crossentropy',
                  optimizer="sgd",
                  metrics=["accuracy", "mse", tensorflow.keras.metrics.Precision(), tensorflow.keras.metrics.Recall()])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    result_dict = history.history
    print(result_dict)

    train_loss = np.average(result_dict["loss"])
    train_accuracy = np.average(result_dict["accuracy"])
    train_mse = np.average(result_dict["mse"])
    train_precision = np.average(result_dict["precision"])
    train_recall = np.average(result_dict["recall"])

    validation_loss = np.average(result_dict["val_loss"])
    validation_accuracy = np.average(result_dict["val_accuracy"])
    validation_mse = np.average(result_dict["val_mse"])
    validation_precision = np.average(result_dict["val_precision"])
    validation_recall = np.average(result_dict["val_recall"])

    score = model.evaluate(x_test, y_test, verbose=0)

    test_loss = score[0]
    test_accuracy = score[1]
    test_mse = score[2]
    test_precision = score[3]
    test_recall = score[4]

    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
    print('test_mse:', test_mse)
    print(score)