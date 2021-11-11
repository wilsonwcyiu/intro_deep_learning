'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''


# from __future__ import print_function
from datetime import datetime

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

from asm2.hyper_para_dict import hyper_para_dict

if __name__ == '__main__':


    batch_size = 128
    num_classes = 10
    epochs = 2

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

    # print(X_train_full.shape)
    # print(X_train_full[0].shape)
    # print(X_train_full.dtype)
    # print(len(X_train_full))
    # exit()
    # print(x_test.shape)
    # exit()

    train_size = 55000
    x_train = X_train_full[-train_size:]
    y_train = y_train_full[-train_size:]

    valid_size = 5000
    x_valid = X_train_full[    :valid_size]
    y_valid = y_train_full[    :valid_size]
    # class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    print(x_train.shape)
    x_train = x_train.reshape(train_size, 784)
    x_train = x_train.astype('float32')
    x_train /= 255

    x_valid = x_valid.reshape(valid_size, 784)
    x_valid = x_valid.astype('float32')
    x_valid /= 255

    test_size = 10000
    x_test = x_test.reshape(test_size, 784)
    x_test = x_test.astype('float32')
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)



    execute_dict_id_list = []#10576
    for idx in range(10000, 10576):
        execute_dict_id_list.append(idx)



    now = datetime.now() # current date and time
    date_time_str: str = now.strftime("%Y%m%d-%H%M%S")
    folder_dir = f"G:/我的雲端硬碟/leiden_university_course_materials/bioinformatics_sem2/introduction_to_deep_learning/asm2/result/"
    result_output_file = folder_dir + date_time_str + "_result_mlp_fashion_output.csv"
    result_backup_output_file = folder_dir + date_time_str + "_result_mlp_fashion_output_backup.csv"
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

    for dict_id in execute_dict_id_list:
        print("\rdict_id: ", dict_id)
        hyper_para = hyper_para_dict[dict_id]

        initializer = hyper_para.initializer
        kernel_regularizer = keras.regularizers.l1_l2(l1=hyper_para.regularization_L1, l2=hyper_para.regularization_L2)
        activation = hyper_para.activation
        dropout = hyper_para.drop_out_rate
        layer_tuple = hyper_para.layer_tuple

        model = Sequential()
        num_of_neuron = layer_tuple[0]
        model.add(Dense(num_of_neuron, activation=activation, input_shape=(784,), kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
        model.add(Dropout(dropout))

        for layer_idx in range(1, len(layer_tuple)):
            num_of_neuron = layer_tuple[layer_idx]
            model.add(Dense(num_of_neuron, activation=activation, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer))
            model.add(Dropout(dropout))

        model.add(Dense(num_classes, activation='softmax'))



        # model.summary()

        loss = hyper_para.loss_func
        optimizer = hyper_para.optimizer
        model.compile(loss=loss,
                      optimizer=optimizer,
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

        # print('Test loss:', test_loss)
        # print('Test accuracy:', test_accuracy)
        # print('test_mse:', test_mse)
        # print(score)


        #store data to report
        dict_id_list.append(dict_id)
        initializer_list.append(             	initializer			           )
        regularization_L1_list.append(       	hyper_para.regularization_L1			                 )
        regularization_L2_list.append(       	hyper_para.regularization_L2			                 )
        activation_list.append(              	activation			          )
        dropout_list.append(                 	dropout			       )
        layer_tuple_list.append(             	layer_tuple			           )
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
                'layer_tuple         ': layer_tuple_list           ,
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
        'layer_tuple         ': layer_tuple_list           ,
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




    # model = keras.models.Sequential([
    #     keras.layers.Flatten(input_shape=[28, 28]),
    #     keras.layers.Dense(300, activation="relu"),
    #     keras.layers.Dense(100, activation="relu"),
    #     keras.layers.Dense(10, activation="softmax")
    # ])
    #
    # model.compile(loss="sparse_categorical_crossentropy",
    #               optimizer="sgd",
    #               metrics=["accuracy"])
    #
    # history = model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))
    #
    #
    # print(history.history)
    #
    #
    # score = model.evaluate(X_test, y_test, verbose=0)

























    # pd.DataFrame(history.history).plot(figsize=(8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    # plt.show()

    exit()
