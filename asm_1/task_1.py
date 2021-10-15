from collections import Counter

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding, TSNE

train_in_file = 'data/train_in.csv'
train_out_file = 'data/train_out.csv'
test_in_file = 'data/test_in.csv'
test_out_file = 'data/test_out.csv'

def main():
    task_1_point_1()
    task_1_point_2()
    task_1_point_3()
    task_1_point_4()

def task_1_point_1():
    mnist_in, mnist_out = load_full_set()
    centers = find_average_digit_representations(mnist_in, mnist_out, plot=False)
    distance_matrix = np.empty((10, 10))
    for i in np.arange(0, 10):
        for j in np.arange(0, 10):
            distance_matrix[i, j] = distance_between_digits(centers[i], centers[j])
    output_matrix_to_latex(distance_matrix, "distances.tex")

def task_1_point_2():
    mnist_in, mnist_out = load_full_set()
    reduced_pca = apply_PCA(mnist_in, 2)
    reduced_dim_plot(reduced_pca, mnist_out, "PCA of MNIST", "MNIST_PCA.png")
    reduced_lle = apply_LLE(mnist_in, 2)
    reduced_dim_plot(reduced_lle, mnist_out, "LLE of MNIST", "MNIST_LLE.png")
    reduced_tsne = apply_tSNE(mnist_in, 2)
    reduced_dim_plot(reduced_tsne, mnist_out, "t-SNE of MNIST", "MNIST_tSNE.png")


def task_1_point_3():
    print("Distance Classifier:")
    train_in, train_out = load_training_set()
    test_in, test_out = load_test_set()
    train_centers = find_average_digit_representations(train_in, train_out)
    train_estimates = classify_digit_set_using_distance_classifier(train_in, train_centers)
    test_estimates = classify_digit_set_using_distance_classifier(test_in, train_centers)
    train_success = evaluate_classification(train_estimates, train_out)
    test_success = evaluate_classification(test_estimates, test_out)
    print("Percentage of correctly classified digits in training set: {:.2f}%".format(train_success * 100))
    print("Percentage of correctly classified digits in test set: {:.2f}%".format(test_success * 100))
    C = calculate_confusion_matrix(test_estimates, test_out)
    output_matrix_to_latex(C, "distance_classifier_confusion_matrix.tex", precision=2)

def task_1_point_4():
    print("kNN Classifier")
    train_in, train_out = load_training_set()
    test_in, test_out = load_test_set()
    full_in, full_out = load_full_set()
    training_eval = []
    test_eval = []
    full_eval = []
    k_values = np.arange(1, 25)
    for k in k_values:
        print("k: {}".format(k))
        train_estimates = classify_digit_set_using_knn_classifier(train_in, k, train_in, train_out)
        test_estimates = classify_digit_set_using_knn_classifier(test_in, k, train_in, train_out)
        full_estimates = classify_digit_set_using_knn_classifier(test_in, k, full_in, full_out)
        train_success = evaluate_classification(train_estimates, train_out)
        training_eval.append(train_success)
        test_success = evaluate_classification(test_estimates, test_out)
        test_eval.append(test_success)
        full_success = evaluate_classification(full_estimates, test_out)
        full_eval.append(full_success)
        print("Percentage of correctly classified digits in training set: {:.2f}%".format(train_success * 100))
        print("Percentage of correctly classified digits in test set: {:.2f}%".format(test_success * 100))
        print("Percentage of correctly classified digits in full set: {:.2f}%".format(full_success * 100))
        C = calculate_confusion_matrix(test_estimates, test_out)
        output_matrix_to_latex(C, "{}-nn_classifier_confusion_matrix.tex".format(k), precision=2)
    fig, ax = plt.subplots()
    ax.plot(k_values, training_eval, label='Training set using Training Set as classifier')
    ax.plot(k_values, test_eval, label='Test set using Training set as classifier')
    ax.plot(k_values, full_eval, label='Test set using Full set as classifier')
    ax.set_title("Fraction of digits correctly classified using KNN")
    ax.set_xlabel("k")
    ax.set_ylabel("Fraction of digits correctly classified")
    ax.legend()
    plt.tight_layout()
    plt.savefig("KNN_eval.png", dpi=300)


def calculate_confusion_matrix(estimated_digits: np.ndarray, true_digits: np.ndarray) -> np.ndarray:
    confusion_matrix = np.empty((10, 10))
    for i in range(0, 10):
        mask_i = (true_digits == i)
        num_i = np.sum(mask_i)
        for j in range(0, 10):
            i_classified_as_j = (estimated_digits[mask_i] == j)
            num_i_classified_as_j = np.sum(i_classified_as_j)
            confusion_matrix[i, j] = num_i_classified_as_j / num_i
    return confusion_matrix


def evaluate_classification(estimated_digits: np.ndarray, true_digits: np.ndarray) -> float:
    num_digits = len(true_digits)
    correctly_classified_digits = (estimated_digits == true_digits)
    return np.sum(correctly_classified_digits) / num_digits


def classify_digit_set_using_knn_classifier(
    digit_set: np.ndarray,
    k: int,
    training_digit_inputs: np.ndarray,
    training_digit_outputs: np.ndarray
) -> np.ndarray:
    classified_digits = []
    for digit in digit_set:
        classified_digits.append(knn_classifier_single_digit(digit, k, training_digit_inputs, training_digit_outputs))
    return np.array(classified_digits)


def knn_classifier_single_digit(
    unkown_digit: np.ndarray,
    k: int,
    training_digit_inputs: np.ndarray,
    training_digit_outputs: np.ndarray
) -> int:
    D = training_digit_inputs - unkown_digit
    distance_to_digits = np.linalg.norm(D, axis=1)
    nearest_neighbour_indices = get_k_smallest_elements(k, distance_to_digits)
    nearest_neighbours = training_digit_outputs[nearest_neighbour_indices]
    return most_frequent(nearest_neighbours)


def get_k_smallest_elements(k, array):
    return np.argsort(array)[1:k+1]

def classify_digit_set_using_distance_classifier(digit_set: np.ndarray, digit_centers: np.ndarray) ->np.ndarray:
    classified_digits = []
    for digit in digit_set:
        classified_digits.append(distance_classifier_single_digit(digit, digit_centers))
    return np.array(classified_digits)


def distance_classifier_single_digit(unknown_digit, digit_centers):
    D = digit_centers - unknown_digit
    distance_to_digit_centers = np.linalg.norm(D, axis=1)
    return np.argmin(distance_to_digit_centers)


def reduced_dim_plot(coords: np.ndarray, classifier: np.ndarray, title: str, filename: str):
    fig, ax = plt.subplots()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colormap = ListedColormap(colors)
    ax.scatter(coords[:, 0], coords[:, 1], c=classifier, cmap=colormap, marker='.')
    """
    The code below is a way to add labels to the pot. We plot each color
    along with each marker at a point outside the range and then limit the
    plot to the range of the data coordinates.
    """
    ax.scatter(1000, 1000, c='tab:blue', label='0')
    ax.scatter(1000, 1000, c='tab:orange', label='1')
    ax.scatter(1000, 1000, c='tab:green', label='2')
    ax.scatter(1000, 1000, c='tab:red', label='3')
    ax.scatter(1000, 1000, c='tab:purple', label='4')
    ax.scatter(1000, 1000, c='tab:brown', label='5')
    ax.scatter(1000, 1000, c='tab:pink', label='6')
    ax.scatter(1000, 1000, c='tab:gray', label='7')
    ax.scatter(1000, 1000, c='tab:olive', label='8')
    ax.scatter(1000, 1000, c='tab:cyan', label='9')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(np.amin(coords[:, 0])*1.05, np.amax(coords[:, 0])*1.05)
    ax.set_ylim(np.amin(coords[:, 1])*1.05, np.amax(coords[:, 1])*1.05)
    # Plot formatting
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def apply_tSNE(data: np.ndarray, dim: int) -> np.ndarray:
    tsne = TSNE(n_components=dim)
    return tsne.fit_transform(data)


def apply_LLE(data: np.ndarray, dim: int) -> np.ndarray:
    lle = LocallyLinearEmbedding(n_components=dim)
    return lle.fit_transform(data)


def apply_PCA(data: np.ndarray, dim: int) -> np.ndarray:
    pca = PCA(n_components=dim)
    return pca.fit_transform(data)


def distance_between_digits(
    digit_representation1: np.ndarray,
    digit_representation2: np.ndarray,
) -> float:
    return np.sqrt(np.sum((digit_representation1 - digit_representation2)**2))


def find_average_digit_representations(
    clouds_in: np.ndarray,
    digit_outputs: np.ndarray,
    plot=False,
) -> np.ndarray:
    centers = np.empty((10, 256))
    for i in np.arange(0, 10):
        digit_clouds = get_all_clouds_for_digit(i, clouds_in, digit_outputs)
        digit_representation = calculate_cloud_center(digit_clouds)
        if plot:
            plot_digit(digit_representation, filename="avg_digits/{}.png".format(i))
        centers[i, :] = digit_representation
    return centers


def output_matrix_to_latex(mat: np.ndarray, filename: str, precision: int=1) -> None:
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df = pd.DataFrame(mat, columns=digits, index=digits)
    format_string = "%.{}f".format(precision)
    tex = df.to_latex(float_format=format_string)
    with open(filename, 'w') as f:
        f.write(tex)


def plot_digit(digit_cloud: np.ndarray, filename=None) -> None:
    fig, ax = plt.subplots()
    im = ax.imshow(np.reshape(digit_cloud, (16, 16)))
    if filename is not None:
        plt.savefig(filename)
    plt.close()


def get_all_clouds_for_digit(
    d: int, 
    clouds_in: np.ndarray,
    digit_outputs: np.ndarray
) -> np.ndarray:
    mask = (digit_outputs == d)
    return clouds_in[mask]


def calculate_cloud_center(cloud_matrix: np.ndarray) -> np.ndarray:
    return np.mean(cloud_matrix, axis=0)


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


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


if __name__ == '__main__':
    main()