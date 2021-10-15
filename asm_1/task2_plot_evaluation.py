import pickle
import numpy as np
from matplotlib import pyplot as plt

import tools
from slp import SLP

diag_file = "randomseed_diagnostics.pkl"

with open(diag_file, "rb") as f:
    data = pickle.load(f)

lr_seed_coordinate = np.array(data[0])
final_test_precision = np.array(data[1])
train_precision_sequence = np.array(data[3])
test_precision_sequence = np.array(data[2])
final_weights_matrix = np.array(data[4])
final_train_precision = train_precision_sequence[:,-1]

mean_test_precision = np.mean(test_precision_sequence, axis=0)
mean_train_precision = np.mean(train_precision_sequence, axis=0)
std_test_precision = np.std(test_precision_sequence, axis=0)
std_train_precision = np.std(train_precision_sequence, axis=0)

best_run_idx = np.argmax(final_test_precision)
worst_run_idx = np.argmin(final_test_precision)

print("Test Set Performance")
print("Average Precision: {:.1f} += {:.1f} %".format(np.mean(final_test_precision), np.std(final_test_precision)))
print("Best Run Precision: {:.1f}%".format(np.amax(final_test_precision)))
print("Worst Run Precision: {:.1f}%".format(np.amin(final_test_precision)))

print("Train Set Performance")
print("Average Precision: {:.1f} += {:.1f} %".format(np.mean(final_train_precision), np.std(final_train_precision)))
print("Best Run Precision: {:.1f}%".format(np.amax(final_train_precision)))
print("Worst Run Precision: {:.1f}%".format(np.amin(final_train_precision)))

epochs = np.arange(0, len(mean_test_precision))

fig, ax = plt.subplots()
ax.plot(epochs, mean_train_precision, label="Training Set", color='tab:blue')
ax.fill_between(
    x=epochs,
    y1=mean_train_precision - std_train_precision,
    y2=mean_train_precision + std_train_precision,
    color='tab:blue',
    alpha=0.1,
    )
ax.plot(epochs, mean_test_precision, label="Test Set", color='tab:orange')
ax.fill_between(
    x=epochs,
    y1=mean_test_precision - std_test_precision,
    y2=mean_test_precision + std_test_precision,
    color='tab:orange',
    alpha=0.1,
    )
ax.plot(epochs, test_precision_sequence[best_run_idx], linestyle='dotted', color='tab:orange')
ax.plot(epochs, train_precision_sequence[best_run_idx], linestyle='dotted', color='tab:blue')

ax.plot(epochs, test_precision_sequence[worst_run_idx], linestyle='dotted', color='tab:orange')
ax.plot(epochs, train_precision_sequence[worst_run_idx], linestyle='dotted', color='tab:blue')
ax.set_title("Learning rate: 0.001")
ax.set_ylim(0, 100)
ax.set_ylabel("Digits classified correctly [%]")
ax.set_xlabel("Number of Training Epochs")
ax.legend()
plt.tight_layout()
plt.savefig("Training_evaluation.png", dpi=300)