import pickle
import numpy as np
from matplotlib import pyplot as plt

import tools
from slp import SLP

def main():
    # Diagnostics
    lr_seed_coordinate = []
    final_test_precision = []
    train_precision_sequence = []
    test_precision_sequence = []
    final_weights_matrix = []


    train_in, train_out = tools.load_training_set()
    test_in, test_out = tools.load_test_set()


    learning_rates = np.logspace(-5, 2, 8)
    # seeds = range(20)
    # num_training_epochs = 5e3
    seeds = [0, 1]
    num_training_epochs = 100


    for lr in learning_rates:
        for seed in seeds:
            slp = SLP(
                learning_rate=lr,
                training_entries_inputs=train_in,
                training_entries_outputs=train_out,
                test_entries_inputs=test_in,
                test_entries_outputs=test_out,
                random_seed=seed,
                )
            slp.train_network_for(num_training_epochs)
            slp.plot_precision_sequence("images/seed={}_lr={}_epochs={}.png".format(seed, lr, num_training_epochs))
            # Save diagnostics
            lr_seed_coordinate.append((lr, seed))
            final_test_precision.append(slp.test_precision[-1])
            train_precision_sequence.append(slp.test_precision)
            test_precision_sequence.append(slp.training_precision)
            final_weights_matrix.append(slp.W)
    with open("diagnostics.pkl", 'wb') as f:
        pickle.dump(
            obj=[
                lr_seed_coordinate, 
                final_test_precision, 
                train_precision_sequence, 
                test_precision_sequence, 
                final_weights_matrix
            ], 
            file=f,
        )

if __name__ == '__main__':
    main()