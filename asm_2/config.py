import os
import numpy as np

clocks_dir = "D:\Data\Clocks"
images = "images.npy"
labels = "labels.npy"

def load_clocks():
	im = np.load(os.path.join(clocks_dir, images))
	l = np.load(os.path.join(clocks_dir, labels))
	return im, l