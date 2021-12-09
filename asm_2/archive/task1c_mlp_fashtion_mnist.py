'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''


# from __future__ import print_function
import tensorflow
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
if __name__ == '__main__':


    batch_size = 128
    num_classes = 10
    epochs = 20

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # print(X_train_full.shape)
    # print(X_train_full[0].shape)
    # print(X_train_full.dtype)
    # print(len(X_train_full))
    # exit()

    X_train = X_train_full[5000:    ] / 255.0
    y_train = y_train_full[5000:    ]

    X_valid = X_train_full[    :5000] / 255.0
    y_valid = y_train_full[    :5000]
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # print(class_names[y_train[0]])


    # model = keras.models.Sequential()
    # model.add(keras.layers.Flatten(input_shape=[28, 28]))
    # model.add(keras.layers.Dense(300, activation="relu"))
    # model.add(keras.layers.Dense(100, activation="relu"))
    # model.add(keras.layers.Dense(10, activation="softmax"))


    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))


    # pd.DataFrame(history.history).plot(figsize=(8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    # plt.show()


    X_new = X_test[:3]
    y_proba = model.predict(X_new)
    print(y_proba.round(2))

    # y_pred = model.predict_classes(X_new)
    y_preds = np.argmax(y_proba,axis=1)
    print(y_preds)
    print(np.array(class_names)[y_preds])
    exit()


    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')
    #
    # # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    # model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(784,)))
    # model.add(Dropout(0.2))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(num_classes, activation='softmax'))
    #
    # model.summary()
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=RMSprop(),
    #               metrics=['accuracy'])
    #
    # history = model.fit(x_train, y_train,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     verbose=1,
    #                     validation_data=(x_test, y_test))
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])