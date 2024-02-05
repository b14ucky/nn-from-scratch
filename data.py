import numpy as np
import pandas as pd


def load_mnist_data():
    with np.load("./dataset/mnist.npz") as data:
        images, labels = data["x_train"], data["y_train"]

    images = images.astype(np.float32) / 255
    images = images.reshape(images.shape[0], (images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]

    return images, labels


def load_my_data():
    data = pd.read_csv("./dataset/numbers.csv", sep=";")
    images, labels = data["Pixels"], data["Labels"]
    images = np.array([np.array(image.split(" "), dtype=np.float32) for image in images]) / 255
    labels = np.eye(10)[labels]

    return images, labels
