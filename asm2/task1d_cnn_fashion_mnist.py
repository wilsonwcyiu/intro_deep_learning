'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import tensorflow
from datetime import datetime

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

from asm2.hyper_para_cnn_dict import hyper_para_cnn_dict
import pandas as pd

if __name__ == '__main__':

    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print(x_train.shape)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    for idx in range(1, 384):
        execute_dict_id_list.append(idx)

    now = datetime.now() # current date and time
    date_time_str: str = now.strftime("%Y%m%d-%H%M%S")
    folder_dir = f"G:/我的雲端硬碟/leiden_university_course_materials/bioinformatics_sem2/introduction_to_deep_learning/asm2/result/"
    result_output_file = folder_dir + date_time_str + "_result_cnn_fashion_output.csv"
    result_backup_output_file = folder_dir + date_time_str + "_result_cnn_fashion_output_backup.csv"
    export_output_interval: int = 20


    dict_id_list: list = []
    initializer_list: list = []
    regularization_L1_list: list = []
    regularization_L2_list: list = []
    activation_list: list = []
    dropout_list: list = []
    network_structure_id_list: list = []
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


    for dict_id in execute_dict_id_list:
        print("\rdict_id: ", dict_id)
        hyper_para = hyper_para_cnn_dict[dict_id]

        initializer = hyper_para.initializer
        kernel_regularizer = keras.regularizers.l1_l2(l1=hyper_para.regularization_L1, l2=hyper_para.regularization_L2)
        activation = hyper_para.activation
        dropout = hyper_para.drop_out_rate


        network_structure_id = hyper_para.network_structure_id
        model = Sequential()
        if network_structure_id == 1:
            model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=input_shape, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            model.add(Conv2D(64, kernel_size=(3, 3), activation=activation, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(dropout))

            model.add(Flatten())

            model.add(Dense(128, activation=activation, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            model.add(Dropout(dropout))

            model.add(Dense(num_classes, activation='softmax'))

        elif network_structure_id == 2:
            model.add(Conv2D(64, kernel_size=(3, 3), activation=activation, input_shape=input_shape, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(dropout))

            model.add(Flatten())

            model.add(Dense(64, activation=activation, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            model.add(Dropout(dropout))

            model.add(Dense(num_classes, activation='softmax'))

        else:
            raise Exception(network_structure_id)

        loss = hyper_para.loss_func
        optimizer = hyper_para.optimizer
        model.compile(loss=loss,  #keras.losses.categorical_crossentropy,
                      optimizer=optimizer, #"sgd", #keras.optimizer.Adadelta(),
                      metrics=["accuracy", "mse", tensorflow.keras.metrics.Precision(), tensorflow.keras.metrics.Recall()])


        epochs = hyper_para.epoch
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))

        result_dict = history.history
        # print(result_dict)

        train_loss = result_dict["loss"][-1]
        train_accuracy = result_dict["accuracy"][-1]
        train_mse = result_dict["mse"][-1]
        # train_precision = result_dict["precision"][-1]
        # train_recall = result_dict["recall"][-1]

        validation_loss = result_dict["val_loss"][-1]
        validation_accuracy = result_dict["val_accuracy"][-1]
        validation_mse = result_dict["val_mse"][-1]
        # validation_precision = result_dict["val_precision"][-1]
        # validation_recall = result_dict["val_recall"][-1]

        score = model.evaluate(x_test, y_test, verbose=0)

        test_loss = score[0]
        test_accuracy = score[1]
        test_mse = score[2]
        # test_precision = score[3]
        # test_recall = score[4]
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



        #store data to report
        dict_id_list.append(dict_id)
        initializer_list.append(             	initializer			           )
        regularization_L1_list.append(       	hyper_para.regularization_L1			                 )
        regularization_L2_list.append(       	hyper_para.regularization_L2			                 )
        activation_list.append(              	activation			          )
        dropout_list.append(                 	dropout			       )
        network_structure_id_list.append(             	network_structure_id			           )
        loss_list.append(                    	loss			    )
        optimizer_list.append(               	optimizer			         )
        epochs_list.append(                  	epochs			      )

        train_loss_list.append(              	train_loss			          )
        train_accuracy_list.append(          	train_accuracy			              )
        train_mse_list.append(               	train_mse			         )
        # train_precision_list.append(         	train_precision			               )
        # train_recall_list.append(            	train_recall			            )
        validation_loss_list.append(         	validation_loss			               )
        validation_accuracy_list.append(     	validation_accuracy			                   )
        validation_mse_list.append(          	validation_mse			              )
        # validation_precision_list.append(    	validation_precision			                    )
        # validation_recall_list.append(       	validation_recall			                 )
        test_loss_list.append(               	test_loss			         )
        test_accuracy_list.append(           	test_accuracy			             )
        test_mse_list.append(                	test_mse			        )
        # test_precision_list.append(          	test_precision			              )
        # test_recall_list.append(             	test_recall			           )


        if dict_id % export_output_interval == 0:
            df = pd.DataFrame({
                'dict id             ': dict_id_list               ,
                'initializer         ': initializer_list           ,
                'regularization_L1   ': regularization_L1_list     ,
                'regularization_L2   ': regularization_L2_list     ,
                'activation          ': activation_list            ,
                'dropout             ': dropout_list               ,
                'network_structure_id         ': network_structure_id           ,
                'loss                ': loss_list                  ,
                'optimizer           ': optimizer_list             ,
                'epochs              ': epochs_list                ,
                'train_loss          ': train_loss_list            ,
                'train_accuracy      ': train_accuracy_list        ,
                'train_mse           ': train_mse_list             ,
                # 'train_precision     ': train_precision_list       ,
                # 'train_recall        ': train_recall_list          ,
                'validation_loss     ': validation_loss_list       ,
                'validation_accuracy ': validation_accuracy_list   ,
                'validation_mse      ': validation_mse_list        ,
                # 'validation_precision': validation_precision_list  ,
                # 'validation_recall   ': validation_recall_list     ,
                'test_loss           ': test_loss_list             ,
                'test_accuracy       ': test_accuracy_list         ,
                'test_mse            ': test_mse_list
                # 'test_precision      ': test_precision_list        ,
                # 'test_recall         ': test_recall_list
            })

            df.to_csv(result_output_file)
            df.to_csv(result_backup_output_file)


        df = pd.DataFrame({
            'dict id             ': dict_id_list               ,
            'initializer         ': initializer_list           ,
            'regularization_L1   ': regularization_L1_list     ,
            'regularization_L2   ': regularization_L2_list     ,
            'activation          ': activation_list            ,
            'dropout             ': dropout_list               ,
            'network_structure_id         ': network_structure_id           ,
            'loss                ': loss_list                  ,
            'optimizer           ': optimizer_list             ,
            'epochs              ': epochs_list                ,
            'train_loss          ': train_loss_list            ,
            'train_accuracy      ': train_accuracy_list        ,
            'train_mse           ': train_mse_list             ,
            # 'train_precision     ': train_precision_list       ,
            # 'train_recall        ': train_recall_list          ,
            'validation_loss     ': validation_loss_list       ,
            'validation_accuracy ': validation_accuracy_list   ,
            'validation_mse      ': validation_mse_list        ,
            # 'validation_precision': validation_precision_list  ,
            # 'validation_recall   ': validation_recall_list     ,
            'test_loss           ': test_loss_list             ,
            'test_accuracy       ': test_accuracy_list         ,
            'test_mse            ': test_mse_list
            # 'test_precision      ': test_precision_list        ,
            # 'test_recall         ': test_recall_list
        })

        df.to_csv(result_output_file)
        df.to_csv(result_backup_output_file)

        print("finish")
        # model = Sequential()
        # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(num_classes, activation='softmax'))
        #
        # model.compile(loss=keras.losses.categorical_crossentropy,
        #               optimizer=keras.optimizer.Adadelta(),
        #               metrics=['accuracy'])
        #
        # model.fit(x_train, y_train,
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           verbose=1,
        #           validation_data=(x_test, y_test))
        # score = model.evaluate(x_test, y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])