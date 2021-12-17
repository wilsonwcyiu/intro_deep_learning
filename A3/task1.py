import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Global Variables
unique_characters = '0123456789+- ' # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 199 # Highest value of integers contained in the queries
max_int_length = len(str(highest_integer)) # 
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = max_int_length + 1    # Maximum length of the answer string

# Savefiles
queryfile = 'queries_nmax={}.npz'.format(highest_integer)

def main():
    X_text, X_img, y_text, y_img = load_query_data()
    X_text_onehot = encode_labels(X_text)
    y_text_onehot = encode_labels(y_text)
    # baseline_prediction(X_text_onehot, y_text_onehot)
    # Train and experiment with the text-to-text RNN Model by using X_text and y_text as your inputs/outputs
    # 1. Try different ratios of train/test splits
    # 2. Try to find more optimal architectures
    text2text = setup_text2text_RNN()
    X_train, X_test, y_train, y_test = train_test_split(
        X_text_onehot, y_text_onehot,
        random_state=13,
        shuffle=True,
        train_size=0.9,
        ) 
    text2text.fit(
        x=X_train, y=y_train,
        validation_data=(X_test, y_test),
        verbose=1,
        )
    choice = np.random.randint(0, len(X_text), size=10)
    entries = X_text[choice]
    print(entries)
    true_answers = y_text[choice] 
    hot_answers = text2text.predict(encode_labels(entries))
    answers = decode_labels(hot_answers)
    for q, a, atrue in zip(entries, answers, true_answers):
        print('{}={}... {}'.format(q, a, atrue))
    import pdb; pdb.set_trace()



def find_best_train_set_size():
    histories = []
    test_set_sizes = np.arange(1, 10) * 0.1
    for ratio in test_set_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X_text_onehot, y_text_onehot,
            random_state=13,
            shuffle=True,
            train_size=ratio,
            )
        text2text = setup_text2text_RNN()
        hist = text2text.fit(
            x=X_train, y=y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            verbose=1,
            )
        histories.append(pd.DataFrame.from_dict(hist.history))
    plot_training_history(histories)

def plot_training_history(histories):
    df = pd.concat(histories, axis=1, keys=test_set_sizes)
    df.columns = df.columns.swaplevel()

    fig, ax = plt.subplots(figsize=(8,5))
    df['loss'].plot(ax=ax, legend=False)
    df['val_loss'].plot(ax=ax, linestyle='--', legend=False)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss")
    ax.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(
        "images/loss_maxint={}.png".format(highest_integer), 
        dpi=300
        )
    plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    df['accuracy'].plot(ax=ax, legend=False)
    df['val_accuracy'].plot(ax=ax, linestyle='--', legend=False)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(
        "images/accuracy_maxint={}.png".format(highest_integer),
        dpi=300
        )

def plot_single_training_history(history, fname):
    h_df = pd.DataFrame.from_dict(history.history)
    fig, ax = plt.subplots()
    ax = h_df.plot(figsize=(8,5))
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Regression Error Measure")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)


def baseline_prediction(X, y):
    y = y.reshape(-1, max_answer_length * len(unique_characters))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(
            input_shape=(max_query_length, len(unique_characters))
            ),
        tf.keras.layers.Dense(max_answer_length * len(unique_characters))
        ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )
    model.summary()
    h = model.fit(
        x=X, y=y,
        epochs=10,
        verbose=0,
        )
    plot_single_training_history(h, 'images/baseline_maxint={}.png'.format(highest_integer))




def setup_text2text_RNN():
    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()
    """
    Encode the input sequence using an RNN, producing an output of size 256.
    In this case the size of our input vectors is [7, 13] as we have queries 
    of length 7 and 13 unique characters. Each of these 7 elements in the 
    query will be fed to the network one by one, as shown in the image above 
    (except with 7 elements).
    Hint: In other applications, where your input sequences have a variable 
    length (e.g. sentences), you would use 
    input_shape=(None, unique_characters).
    """
    text2text.add(
        tf.keras.layers.LSTM(
            units=256,
            input_shape=(max_query_length, len(unique_characters))
            )
        )
    """
    As the decoder RNN's input, repeatedly provide with the last output of RNN
    for each time step. Repeat 4 times as that's the maximum length of the 
    output (e.g. '  1-199' = '-198') when using 3-digit integers in queries. 
    In other words, the RNN will always produce 4 characters as its output.
    """
    text2text.add(
        tf.keras.layers.RepeatVector(max_answer_length)
        )
    """
    By setting return_sequences to True, return not only the last output but
    all the outputs so far in the form of (num_samples, timesteps, output_dim).
    This is necessary as TimeDistributed in the below expects the first
    dimension to be the timesteps.
    """
    text2text.add(
        tf.keras.layers.LSTM(
            units=128,
            return_sequences=True
            )
        )
    """
    Apply a dense layer to the every temporal slice of an input. For each of
    step of the output sequence, decide which character should be chosen.
    """
    text2text.add(
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(len(unique_characters), activation='softmax')
            )
        )
    """
    Next we compile the model using categorical crossentropy as our loss
    function.
    """
    text2text.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )
    text2text.summary()
    return text2text

def generate_images(cross=False, n=50):
    """
    Creates 'n' images of randm minus and plus signs. First create empty images
    then draw either one (minus) or two (plus) lines using 'cv2.line()'
    See https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#cv2.line
    Inputs:
        cross - If True, draw plus signs, draw minus otherwise
        n - number of signs to create
    Returns:
        A np.ndarray op dim (n, 28, 28) containing n 28x28 images of the 
        specified sign.
    """
    import cv2

    x = np.random.randint(12, 16, (n, 2))
    y1 = np.random.randint(4, 8, n)
    y2 = np.random.randint(20, 24, n)
    
    blank = np.zeros([n, 28, 28])
    for i in range(n):
        line = cv2.line(
            img=blank[i],
            pt1=(y1[i], x[i,0]),
            pt2=(y2[i], x[i, 1]),
            color=(255,0,0),
            thickness=2,
            lineType =cv2.LINE_AA
            )
        if cross:
            line = cv2.line(
                img=blank[i],
                pt1=(x[i,0], y1[i]),
                pt2=(x[i, 1], y2[i]),
                color=(255,0,0),
                thickness=2,
                lineType =cv2.LINE_AA
                )
    return blank

def save_generated(images, fname):
    for i in range(20):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

def create_data(highest_integer):
    """
    Creates the following data for all pairs of integers up to [highest integer]+[highest_integer]:

    @return:
    X_text: '151+ 21' -> text query of an arithmetic operation
    X_img : Stack of MNIST images corresponding to the query (7 x 28 x 28)
    y_text: ' 172' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (4 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """
    print("Creating Data")
    (MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(cross=True)
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []
    for i in range(highest_integer + 1):
        for j in range(highest_integer + 1):
            
            i_char = to_padded_chars(i, max_len=max_int_length)
            j_char = to_padded_chars(j, max_len=max_int_length)

            for sign in ['-', '+']:
                query_string = i_char + sign + j_char
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))
            
    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.
  
def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)

def display_sample(n, fname):
    labs = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        plt.axis('off')
        plt.title(labs[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nSample ID: {n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

def encode_labels(labels, max_len=4):
    n = len(labels)
    length = len(labels[0])
    char_map = dict(zip(unique_characters, range(len(unique_characters))))
    one_hot = np.zeros([n, length, len(unique_characters)])
    for i, label in enumerate(labels):
        m = np.zeros([length, len(unique_characters)])
        for j, char in enumerate(label):
            m[j, char_map[char]] = 1
        one_hot[i] = m

    return one_hot 

def decode_labels(labels):
    pred = np.argmax(labels, axis=2)
    decoded_predictions = []
    for pred_i in pred:
        seq = ''.join([unique_characters[i] for i in pred_i])
        decoded_predictions.append(seq)
    return decoded_predictions

def load_query_data():
    try:
        with open(queryfile, 'rb') as f:
            data = np.load(f)
            X_text = data['arr_0']
            X_img = data['arr_1']
            y_text = data['arr_2']
            y_img = data['arr_3']
    except:
        X_text, X_img, y_text, y_img = create_data(highest_integer)
        with open(queryfile, 'wb') as f:
            np.savez(f, X_text, X_img, y_text, y_img)
    return X_text, X_img, y_text, y_img

if __name__ == '__main__':
    main()