import numpy as np
import pandas as pd
import argparse
import cv2


class DataLoader:
    @staticmethod
    def load_mnist_data():
        with np.load("./dataset/mnist.npz") as data:
            images, labels = data["x_train"], data["y_train"]

        images = images.astype(np.float32) / 255
        images = images.reshape(images.shape[0], (images.shape[1] * images.shape[2]))
        labels = np.eye(10)[labels]

        # split the data into training and test sets
        split = int(0.8 * images.shape[0])
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
                cv2.line(self.image, (self.x, self.y), (x, y), color=(255, 255, 255), thickness=40)
                self.x, self.y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.image, (self.x, self.y), (x, y), color=(255, 255, 255), thickness=40)

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
