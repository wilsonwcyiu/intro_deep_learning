from copy import deepcopy
import time
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(42)

import tools

def main():
    test_weights = SingleLayerPerceptron._initiate_weights_matrix(257, 10)
    train_in, train_out = tools.load_training_set()
    test_in, test_out = tools.load_test_set()
    learning_rates = [0.01, 0.03, 0.05, 0.07, 0.09]
    for lr in learning_rates:
        slp = SingleLayerPerceptron(
            learning_rate=lr,
            training_entries_inputs=train_in,
            training_entries_outputs=train_out,
            test_entries_inputs=test_in,
            test_entries_outputs=test_out,
            weights_matrix=deepcopy(test_weights),
        )
        slp.train_network(int(3e5))
        result = slp.classify(test_in)
        successes = (result == test_out)
        successful_classification_fraction = np.sum(successes) / len(test_out)
        print("Test Classification succes = {}%.".format(successful_classification_fraction * 100))
        slp.plot_precision_sequence("images/lr={}_epochs={}.png".format(lr, int(3e5)))


class SingleLayerPerceptron:

    def __init__(
        self,
        learning_rate: float,
        training_entries_inputs: np.ndarray,
        training_entries_outputs: np.ndarray,
        test_entries_inputs: np.ndarray,
        test_entries_outputs: np.ndarray,
        weights_matrix = None,
    ):
        assert len(training_entries_inputs) == len(training_entries_outputs)
        assert len(test_entries_inputs) == len(test_entries_outputs)
        assert training_entries_inputs.shape[1] == test_entries_inputs.shape[1]

        self.num_nodes = 10
        self.learning_rate = learning_rate
        self.training_input_matrix = self._append_bias_to_input_matrix(training_entries_inputs)
        self.training_desired_outputs = training_entries_outputs
        self.training_desired_activation_matrix = SingleLayerPerceptron._convert_output_digits_to_desired_node_activation_matrix(training_entries_outputs, self.num_nodes)
        self.test_input_matrix = self._append_bias_to_input_matrix(test_entries_inputs)
        self.test_desired_outputs = test_entries_outputs
        self.test_desired_activation_matrix = SingleLayerPerceptron._convert_output_digits_to_desired_node_activation_matrix(test_entries_outputs, self.num_nodes)
        self.num_entries = len(self.training_input_matrix)
        self.num_inputs = len(self.training_input_matrix[1])
        if weights_matrix is None:
            self.W = self._initiate_weights_matrix(self.num_inputs, self.num_nodes)
        else:
            assert weights_matrix.shape == (self.num_inputs, self.num_nodes)
            self.W = weights_matrix
            print(self.W)
        self.training_current_activations = None
        self.test_current_activations = None
        self.training_precision = []
        self.test_precision = []


    def train_network(self, training_epochs=5000):
        print("Training Network")
        print("Learning Rate: {}".format(self.learning_rate))
        print("Training Epochs: {}".format(training_epochs))
        start = time.time()
        for epoch in range(training_epochs):
            self.training_current_activations, self.training_output = SingleLayerPerceptron.forward_pass(
                X = self.training_input_matrix,
                W = self.W,
            )
            current_epoch_training_error_matrix, W_updates = SingleLayerPerceptron.backward_pass(
                learning_rate = self.learning_rate,
                X = self.training_input_matrix,
                A = self.training_current_activations,
                D = self.training_desired_activation_matrix,
            )
            self.W += W_updates
            self.training_precision.append(self.calculate_precision_on_training_set())
            self.test_precision.append(self.calculate_precision_on_test_set())
        stop = time.time()
        print("Time elapsed: {:.1f}s".format(stop-start))


    def classify(self, input_matrix):
        input_matrix_with_bias = SingleLayerPerceptron._append_bias_to_input_matrix(input_matrix)
        A, classifications = SingleLayerPerceptron.forward_pass(input_matrix_with_bias, self.W)
        return classifications


    def calculate_precision_on_training_set(self):
        successes = (self.training_output == self.training_desired_outputs)
        return np.sum(successes) / len(self.training_desired_outputs) * 100


    def calculate_precision_on_test_set(self):
        activation_matrix, predictions = SingleLayerPerceptron.forward_pass(
            X = self.test_input_matrix,
            W = self.W
        )
        successes = (predictions == self.test_desired_outputs)
        return np.sum(successes) / len(self.test_desired_outputs) * 100


    def plot_precision_sequence(self, filename=None):
        fig, ax = plt.subplots()
        trained_epochs = len(self.training_precision)
        ax.plot(range(trained_epochs), self.training_precision, label="Training Set")
        ax.plot(range(trained_epochs), self.test_precision, label="Test Set")
        ax.set_xscale('log')
        ax.set_title("Learning rate: {}".format(self.learning_rate))
        ax.set_ylabel("Digits classified correctly [%]")
        ax.set_xlabel("Number of Training Epochs")
        ax.legend()
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


    @staticmethod
    def test_convergence(precision_sequence, final_num_epochs):
        final_epochs = precision_sequence[-final_num_epochs:]
        if len(precision_sequence) > final_num_epochs:
            decreasing = all(i < j for i, j in zip(final_epochs, final_epochs[1:]))
            # similar = (np.std(precision_sequence[-final_num_epochs:]) < np.mean(precision_sequence[-final_num_epochs:]))
            if decreasing:
                return True
            else:
                return False
        else:
            return False


    @staticmethod
    def forward_pass(X, W):
        A = SingleLayerPerceptron.calculate_activations(X, W)
        predictions = SingleLayerPerceptron.determine_outputs(A)
        return A, predictions


    @staticmethod
    def backward_pass(learning_rate, X, A, D):
        dPHI = SingleLayerPerceptron._sigmoid_der(A)
        E = D - A
        dW = learning_rate * np.matmul(X.T, E*dPHI)
        return E, dW


    @staticmethod
    def calculate_activations(X, W):
        net_activation = np.matmul(X, W)
        return SingleLayerPerceptron._sigmoid(net_activation)


    @staticmethod
    def determine_outputs(A):
        return np.argmax(A, axis=1)


    @staticmethod
    def _append_bias_to_input_matrix(input_matrix):
        num_entries = input_matrix.shape[0]
        biases = np.ones((num_entries, 1))
        return np.concatenate((input_matrix, biases), axis=1)


    @staticmethod
    def _convert_output_digits_to_desired_node_activation_matrix(output_list, num_nodes):
        num_entries = len(output_list)
        desired_activations = np.zeros((num_entries, num_nodes))
        # The magic slicing term below gives two lists of coordinates to the
        # array. The first list 'np.arange(num_entries)' denotes we want to
        # access each row. The second list 'output_list' species the column
        # index for each row.
        desired_activations[np.arange(num_entries), output_list] = 1
        return desired_activations


    @staticmethod
    def _initiate_weights_matrix(num_inputs, num_nodes):
        return np.random.uniform(
            low=-1,
            high=1,
            size=(num_inputs, num_nodes)
        )


    @staticmethod
    def _sigmoid(z):
        return 1/(1 + np.exp(-z))


    @staticmethod
    def _sigmoid_der(sigmoid_z):
        return sigmoid_z * (1-sigmoid_z)


if __name__ == '__main__':
    main()