'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

from datetime import datetime

from tensorflow import keras

from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K



if __name__ == '__main__':

    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)



    execute_dict_id_list = []
    for idx in range(10000, 10576):
        execute_dict_id_list.append(idx)

    now = datetime.now() # current date and time
    date_time_str: str = now.strftime("%Y%m%d-%H%M%S")
    folder_dir = f"G:/我的雲端硬碟/leiden_university_course_materials/bioinformatics_sem2/introduction_to_deep_learning/asm2/result/"
    result_output_file = folder_dir + date_time_str + "_result_mlp_mnist_output.csv"
    result_backup_output_file = folder_dir + date_time_str + "_result_mlp_mnist_output_backup.csv"
    export_output_interval: int = 20


    dict_id_list: list = []
    initializer_list: list = []
    regularization_L1_list: list = []
    regularization_L2_list: list = []
    activation_list: list = []
    dropout_list: list = []
    layer_tuple_list: list = []
    loss_list: list = []
    optimizer_list: list = []
    epochs_list: list = []
    train_loss_list: list = []
    train_accuracy_list: list = []
    train_mse_list: list = []
    train_precision_list: list = []
    train_recall_list: list = []
    validation_loss_list: list = []
    validation_accuracy_list: list = []
    validation_mse_list: list = []
    validation_precision_list: list = []
    validation_recall_list: list = []
    test_loss_list: list = []
    test_accuracy_list: list = []
    test_mse_list: list = []
    test_precision_list: list = []
    test_recall_list: list = []




    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="sgd", #keras.optimizer.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])