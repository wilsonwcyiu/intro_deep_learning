from collections import Counter

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE


train_in_file = 'data/train_in.csv'
train_out_file = 'data/train_out.csv'
test_in_file = 'data/test_in.csv'
test_out_file = 'data/test_out.csv'


def load_full_set():
    train_in, train_out = load_training_set()
    test_in, test_out = load_test_set()
    full_in = np.concatenate((test_in, train_in), axis=0)
    full_out = np.concatenate((test_out, train_out), axis=0)
    return full_in, full_out


def load_training_set():
    train_in = load(train_in_file)
    train_out = load(train_out_file).astype('int')
    return train_in, train_out


def load_test_set():
    test_in = load(test_in_file)
    test_out = load(test_out_file).astype('int')
    return test_in, test_out


def load(filename: str) -> np.ndarray:
    return np.genfromtxt(filename, delimiter=',')


def get_most_frequent_item(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def get_k_smallest_elements(k, array):
    return np.argsort(array)[1:k+1]


def apply_tSNE(data: np.ndarray, output_dimension: int) -> np.ndarray:
    tsne = TSNE(n_components=output_dimension)
    return tsne.fit_transform(data)


def apply_LLE(data: np.ndarray, output_dimension: int) -> np.ndarray:
    lle = LocallyLinearEmbedding(n_components=output_dimension)
    return lle.fit_transform(data)


def apply_PCA(data: np.ndarray, output_dimension: int) -> np.ndarray:
    pca = PCA(n_components=output_dimension)
    return pca.fit_transform(data)