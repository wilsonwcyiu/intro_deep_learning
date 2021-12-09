"""
Assignment 2
Introduction to Deep Learning (2021 Fall Semester, Universiteit Leiden)

Hardware used:
CPU: 4x Intel Core i5-5470 CPU @ 3.20 GHz
GPU: NVIDIA GeForce GTX 750 Ti
RAM: 8.00 GB
"""
import os

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import config

np.random.seed(17)

testing_mode = False
# Hyperparams
epochs = 8
lr = 0.01
# Loss Function
loss_fn = "mean_absolute_error"
# Optimizer
opt = keras.optimizers.SGD(
    learning_rate=lr,
    momentum=0.0,
    nesterov=False,
    # name="mySGD",
)
# Metrics
metrics = [
	tf.keras.metrics.MeanSquaredError(),
	]

def main():
	images, labels = config.load_clocks()
	num_clocks = images.shape[0]
	labels_normalised_float = time_to_float(labels) / 12
	train, test, validate = train_test_validate_indices(num_clocks)
	train_X = images[train]
	# Here reshape is called to transform the arrays from (N,) to (N,1)
	train_y = labels_normalised_float[train].reshape(-1, 1)
	print("Train Shape: ", train_X.shape, train_y.shape)
	print("Train Size: ", train_X.size, train_y.size)
	print("Train dtype: ", train_X.dtype, train_y.dtype)
	test_X = images[test]
	test_y = labels_normalised_float[test].reshape(-1, 1)
	validate_X = images[validate]
	validate_y = labels_normalised_float[validate].reshape(-1, 1)
	if testing_mode is True:
		# If we want to quickly test some changes to the network
		train_X, train_y = reduce_set_size_to(10, train_X, train_y)
		test_X, test_y = reduce_set_size_to(10, test_X, test_y)
		validate_X, validate_y = reduce_set_size_to(10, validate_X, validate_y)
	m, h, s = simple_CNN(train_X, train_y, validate_X, validate_y, test_X, test_y)
	plot_training_history(h)

def plot_training_history(history):
	h_df = pd.DataFrame.from_dict(history.history)
	fig, ax = plt.subplots()
	ax = h_df.plot(figsize=(8,5))
	ax.set_xlabel("Training Epoch")
	ax.set_ylabel("Regression Error Measure")
	plt.tight_layout()
	plt.savefig("regression_CNN_lr={}.png".format(lr))


def simple_CNN(x_train, y_train, x_validate, y_validate, x_test, y_test):
	model = keras.models.Sequential([
		keras.layers.Conv2D(filters=64, kernel_size=10, strides=(3,3),
							padding="same", activation="relu",
							input_shape=[150, 150, 1]),
		keras.layers.MaxPooling2D(2),
		keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1),
							padding="same", activation="relu"),
		keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1),
							padding="same", activation="relu"),
		keras.layers.MaxPooling2D(2),
		keras.layers.Conv2D(filters=256, kernel_size=2, strides=(1,1),
							padding="same", activation="relu"),
		keras.layers.Conv2D(filters=256, kernel_size=2, strides=(1,1),
							padding="same", activation="relu"),
		keras.layers.MaxPooling2D(2),
		keras.layers.Flatten(),
		keras.layers.Dense(128, activation="relu"),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(64, activation="relu"),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(1, activation="sigmoid"),
		])
	model.compile(
		loss=loss_fn,
		optimizer=opt,
		metrics=metrics,
	)
	print(model.summary())
	history = model.fit(
		x_train, y_train,
		batch_size=32,
		epochs=epochs,
		verbose=1,
		validation_data=(x_validate, y_validate)
	)
	score = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	return model, history, score

def reduce_set_size_to(percent:int, X, y):
	size = len(X)
	new_size = round(size * percent / 100)
	return X[:new_size], y[:new_size]

def train_test_validate_indices(
			num_indices: int,
			set_fractions=[0.7, 0.2, 0.1],
		):
	"""
	Returns three lists of indices for the training, test and validation
	set. The ratios of the sets are given by 'set_fractions', a list of three
	numbers that should sum to one. Indices are shuffled and then divided into
	parts. 
	"""
	assert len(set_fractions) == 3
	assert round(np.sum(set_fractions)) == 1
	permute = np.random.permutation(num_indices)
	train_end_idx = round(num_indices * set_fractions[0])
	test_end_idx = round(num_indices * (set_fractions[0] + set_fractions[1]))
	train = permute[:train_end_idx]
	test = permute[train_end_idx: test_end_idx]
	validate = permute[test_end_idx:]
	return train, test, validate

def time_diff_loss_fn(y_true, y_pred):
	# TODO get this working
	diff1 = tf.math.abs(y_true - y_pred)
	diff2 = 1 - diff1
	final_diff = tf.math.minimum(diff1, diff2)
	return final_diff


def time_diff(y_true: np.ndarray, y_pred: np.ndarray):
	if not y_true.dtype == float:
		xf = time_to_float(y_true)
	else:
		xf = y_true
	if not y_pred.dtype == float:
		yf = time_to_float(y_pred)
	else:
		yf = y_pred
	diff1 = np.abs(xf - yf)
	diff2 = 12 - diff1	
	return np.minimum(diff1, diff2)

def time_to_float(hhmm: np.ndarray):
	hh = hhmm[:, 0]
	mm = hhmm[:, 1]
	f = hh + mm / 60
	return f

def float_to_time(f: np.ndarray):
	hh = f.astype(int)
	mm = np.round((f - hh) * 60)
	return np.transpose(np.array([hh, mm], dtype=np.int8))

def assert_time_conversion_scheme(timestamps):
	i = np.random.randint(low=0, high=timestamps.shape[0], size=1000)
	ftime = time_to_float(timestamps[i])
	time = float_to_time(ftime)
	assert np.all(timestamps[i] == time)

def plot_a_few_random_clocks(images, few=10):
	img_indices = np.random.randint(low=0, high=images.shape[0], size=few)
	for i in img_indices:
		plt.imshow(images[i, :, :], cmap='gray', vmin=0, vmax=255)
		filename = os.path.join(config.clocks_dir, "{:05d}.png".format(i))
		plt.savefig(filename)
		plt.close()

if __name__ == '__main__':
	main()