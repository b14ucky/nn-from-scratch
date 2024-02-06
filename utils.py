import numpy as np
import pandas as pd
import argparse


class DataLoader:
    @staticmethod
    def load_mnist_data():
        with np.load("./dataset/mnist.npz") as data:
            images, labels = data["x_train"], data["y_train"]

        images = images.astype(np.float32) / 255
        images = images.reshape(images.shape[0], (images.shape[1] * images.shape[2]))
        labels = np.eye(10)[labels]

        return images, labels

    @staticmethod
    def load_custom_data():
        data = pd.read_csv("./dataset/numbers.csv", sep=";").sample(frac=1).reset_index(drop=True)
        images, labels = data["Pixels"], data["Labels"]
        images = np.array([np.array(image.split(" "), dtype=np.float32) for image in images]) / 255
        labels = np.eye(10)[labels]

        return images, labels


class EpochAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values < 1:
            raise argparse.ArgumentError(self, "Epochs must be greater than 0")
        setattr(namespace, self.dest, values)


class LearningRateAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values <= 0 or values >= 1:
            raise argparse.ArgumentError(self, "Learning rate must be between 0 and 1")
        setattr(namespace, self.dest, values)
