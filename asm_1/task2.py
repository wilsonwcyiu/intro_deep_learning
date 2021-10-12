import numpy as np
np.random.seed(42)

import tools

def main():
    train_in, train_out = tools.load_training_set()
    test_in, test_out = tools.load_test_set()
    slp = SingleLayerPerceptron(train_in, train_out, test_in, test_out)
    slp.train_network()


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

        self.training_input_matrix = self._append_bias_to_input_matrix(training_entries_inputs)
        self.training_outputs = training_entries_output
        self.test_input_matrix = self._append_bias_to_input_matrix(test_entries_inputs)
        self.test_outputs = test_entries_outputs

        self.num_nodes = 10
        self.num_entries = len(self.training_input_matrix)
        self.num_inputs = len(self.training_input_matrix[1])

        self.W = self._initiate_weights_matrix(self.num_inputs, self.num_nodes)


    def train_network(self):
        self.forward_pass()
        

    def forward_pass(self):
        self.current_activations = SingleLayerPerceptron.calculate_activations(
            self.training_input_matrix, 
            self.W
            )
        self.current_outputs = SingleLayerPerceptron.determine_outputs(
            self.current_activations
            )


    @staticmethod
    def calculate_activations(T, W):
        activation_matrix = np.matmul(T, W)
        return SingleLayerPerceptron._sigmoid(activation_matrix)


    @staticmethod
    def determine_outputs(A):
        return np.argmax(A, axis=1)


    @staticmethod
    def _append_bias_to_input_matrix(input_matrix):
        num_entries = input_matrix.shape[0]
        biases = np.ones((num_entries, 1))
        return np.concatenate((input_matrix, biases), axis=1)


    @staticmethod
    def _initiate_weights_matrix(num_inputs, num_nodes):
        return np.random.rand(num_inputs, num_nodes)


    @staticmethod
    def _sigmoid(x):
        return 1/(1 + np.exp(-x))


    @staticmethod
    def _sigmoid_der(x):
        return sigmoid(x)*(1-sigmoid(x))

if __name__ == '__main__':
    main()