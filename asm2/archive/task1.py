from sklearn.datasets import fetch_openml

import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':

    print("start")
    mnist = fetch_openml('mnist_784', version=1, data_home="D:/tmp_data/", as_frame=False)

    print("end get data")

    print(mnist.keys())

    X, y = mnist["data"], mnist["target"]
    print(type(X))
    print(X.shape)
    print(y.shape)

    print(X[1])

    some_digit = X[0]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()