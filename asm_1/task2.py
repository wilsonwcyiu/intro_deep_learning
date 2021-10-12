import numpy as np

import tools

def main():
    train_in, train_out = tools.load_training_set()
    test_in, test_out = tools.load_test_set()
    train_in = append_bias_to_inputs(train_in)
    test_in = append_bias_to_inputs(test_in)

def append_bias_to_inputs(input_set):
    num_entries = input_set.shape[0]
    biases = np.ones((num_entries, 1))
    return np.concatenate((input_set, biases), axis=1)

if __name__ == '__main__':
    main()