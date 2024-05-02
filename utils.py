import numpy as np
import pandas as pd
import argparse
import cv2
import matplotlib.pyplot as plt
from IPython import display
from numpy.typing import NDArray


class DataLoader:
    @staticmethod
    def load_mnist_data():
        with np.load("./dataset/mnist.npz") as data:
            images, labels = data["x_train"], data["y_train"]

        images = images.astype(np.float32) / 255
        images = images.reshape(images.shape[0], (images.shape[1] * images.shape[2]))
        labels = np.eye(10)[labels]

        # split the data into training and test sets
        split = int(0.9 * images.shape[0])
        train_images, test_images = images[:split], images[split:]
        train_labels, test_labels = labels[:split], labels[split:]

        return train_images, train_labels, test_images, test_labels

    @staticmethod
    def load_custom_data():
        data = pd.read_csv("./dataset/numbers.csv", sep=";").sample(frac=1).reset_index(drop=True)
        images, labels = data["Pixels"], data["Labels"]
        images = np.array([np.array(image.split(" "), dtype=np.float32) for image in images]) / 255
        labels = np.eye(10)[labels]

        # split the data into training and test sets
        split = int(0.8 * images.shape[0])
        train_images, test_images = images[:split], images[split:]
        train_labels, test_labels = labels[:split], labels[split:]

        return train_images, train_labels, test_images, test_labels


class NonpositiveIntAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values < 1:
            raise argparse.ArgumentError(self, "Value should be greater than 0")
        setattr(namespace, self.dest, values)


class LearningRateAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values <= 0 or values >= 1:
            raise argparse.ArgumentError(self, "Learning rate must be between 0 and 1")
        setattr(namespace, self.dest, values)


class DrawingApp:
    def __init__(self):
        self.drawing = False
        self.x, self.y = None, None
        self.image = np.zeros((512, 512, 1), np.float32)
        self.window_name = "Draw here!"
        self.running = True

    def _reset(self):
        self.image = np.zeros((512, 512, 1), np.float32)
        self.x, self.y = None, None
        self.drawing = False

    def _line_drawing(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x, self.y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.image, (self.x, self.y), (x, y), color=(255, 255, 255), thickness=35)
                self.x, self.y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.image, (self.x, self.y), (x, y), color=(255, 255, 255), thickness=35)

    def get_image(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._line_drawing)

        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1)
            if key == ord("r"):
                self._reset()
            elif key == 13:
                break
            elif key == 27:
                self.running = False
                return None
        cv2.destroyAllWindows()

        image = cv2.resize(self.image, (28, 28))
        image = image.flatten() / 255

        self._reset()
        return image


def plot(loss, test_accuracy, train_accuracy):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.plot(loss, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracy, label="Test Accuracy")
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.legend()

    plt.show(block=False)
    plt.pause(0.1)


class ImageAugmentation:
    def __init__(self, image: NDArray):
        self.image: NDArray = image.reshape(28, 28) * 255
        self.h, self.w = self.image.shape

    def __call__(self) -> NDArray:
        random_zoom: float = np.random.uniform(1, 1.15)
        self.image = self._zoom(random_zoom)
        random_rotation: int = np.random.randint(-10, 10)
        self.image = self._rotate(random_rotation)
        random_shift_x: int = np.random.randint(-3, 3)
        random_shift_y: int = np.random.randint(-3, 3)
        self.image = self._shift(random_shift_x, random_shift_y)
        random_noise: float = np.random.uniform(0, 20)
        self.image = self._noise(random_noise)
        return self.image.reshape(784) / 255

    def _zoom(self, scale: float) -> NDArray:
        h_new, w_new = int(self.h * scale), int(self.w * scale)
        img_new: NDArray = cv2.resize(self.image, (w_new, h_new))

        if h_new < self.h:
            pad_top = (self.h - h_new) // 2
            pad_bottom = self.h - h_new - pad_top
            img_new = np.pad(img_new, ((pad_top, pad_bottom), (0, 0)), mode="constant")
        elif h_new > self.h:
            img_new = img_new[(h_new - self.h) // 2 : (h_new + self.h) // 2, :]

        if w_new < self.w:
            pad_left = (self.w - w_new) // 2
            pad_right = self.w - w_new - pad_left
            img_new = np.pad(img_new, ((0, 0), (pad_left, pad_right)), mode="constant")
        elif w_new > self.w:
            img_new = img_new[:, (w_new - self.w) // 2 : (w_new + self.w) // 2]

        return img_new

    def _rotate(self, angle: float) -> NDArray:
        M: NDArray = cv2.getRotationMatrix2D(((self.w - 1) / 2, (self.h - 1) / 2), angle, 1)
        img_new: NDArray = cv2.warpAffine(self.image, M, (self.w, self.h))
        return img_new

    def _shift(self, dx: int, dy: int) -> NDArray:
        M: NDArray = np.float32([[1, 0, dx], [0, 1, dy]])
        img_new: NDArray = cv2.warpAffine(self.image, M, (self.w, self.h))
        return img_new

    def _noise(self, sigma: float) -> NDArray:
        noise: NDArray = np.random.normal(0, sigma, self.image.shape)
        img_new: NDArray = self.image + noise
        img_new = np.clip(img_new, 0, 255)
        return img_new
