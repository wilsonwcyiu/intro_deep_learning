import numpy as np
np.random.seed(42)

import tools

def main():
    train_in, train_out = tools.load_training_set()
    test_in, test_out = tools.load_test_set()
    slp = SingleLayerPerceptron(train_in, train_out, test_in, test_out)


class SingleLayerPerceptron:

    def __init__(
        self,
        training_entries_inputs: np.ndarray,
        training_entries_output: np.ndarray,
        test_entries_inputs: np.ndarray,
        test_entries_outputs: np.ndarray,
    ):
        assert len(training_entries_inputs) == len(training_entries_output)
        assert len(test_entries_inputs) == len(test_entries_outputs)
        assert training_entries_inputs.shape[1] == test_entries_inputs.shape[1]

        self.num_nodes = 10
        self.num_entries = len(training_entries_inputs)
        self.num_inputs = len(training_entries_inputs[1])

        self.training_input_matrix = self._append_bias_to_input_matrix(training_entries_inputs)
        self.training_outputs = training_entries_output
        self.test_input_matrix = self._append_bias_to_input_matrix(test_entries_inputs)
        self.test_outputs = test_entries_outputs


    def _append_bias_to_input_matrix(self, input_matrix):
        num_entries = input_matrix.shape[0]
        biases = np.ones((num_entries, 1))
        return np.concatenate((input_matrix, biases), axis=1)


    def _initiate_weights_matrix(self):
        pass


if __name__ == '__main__':
    main()