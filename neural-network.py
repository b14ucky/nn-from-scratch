import numpy as np
import matplotlib.pyplot as plt
from utils import DataLoader, EpochAction, LearningRateAction, DrawingApp
import argparse


class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = "sigmoid") -> None:
        """
        Args:
            input_size (int): The number of input nodes
            output_size (int): The number of output nodes
            activation (str, optional): The activation function to use. Defaults to "sigmoid".
        """
        # initialize the weights with he-initialization
        self.weights: np.ndarray = np.random.uniform(
            0, np.sqrt(2 / input_size), (output_size, input_size)
        )
        self.bias: np.ndarray = np.zeros((output_size, 1))
        self.activation: str = activation

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Overload the call operator to make the layer callable
        Args:
            input (np.ndarray): The input to the layer
        Returns:
            np.ndarray: The output of the layer
        """
        return self._forward(input)

    def _forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer
        Args:
            input (np.ndarray): The input to the layer
        Returns:
            np.ndarray: The output of the layer
        """
        self.input: np.ndarray = input
        self.output: np.ndarray = self.weights @ self.input + self.bias
        if self.activation == "sigmoid":
            return self._sigmoid(self.output)
        elif self.activation == "relu":
            return self._relu(self.output)

    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the layer
        Args:
            gradient (np.ndarray): The gradient of the loss with respect to the output of the layer
            learning_rate (float): The learning rate to use for updating the weights and bias
        Returns:
            np.ndarray: The gradient of the loss with respect to the input of the layer
        """
        # calculate the delta l
        if self.activation == "sigmoid":
            delta: np.ndarray = gradient * self._dsigmoid(self.output)
        elif self.activation == "relu":
            delta: np.ndarray = gradient * self._drelu(self.output)

        # calculate the gradients with respect to the weights and bias
        self.weights_gradient: np.ndarray = delta @ np.transpose(self.input)
        self.bias_gradient: np.ndarray = delta

        # update the weights and bias
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient

        # calculate and return the delta l-1
        return np.transpose(self.weights) @ delta

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        The sigmoid activation function
        Args:
            x (np.ndarray): The input to the activation function
        Returns:
            np.ndarray: The output of the activation function
        """
        return 1 / (1 + np.exp(-x))

    def _dsigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        The derivative of the sigmoid activation function
        Args:
            x (np.ndarray): The input to the activation function
        Returns:
            np.ndarray: The output of the derivative of the activation function
        """
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """
        The ReLU activation function
        Args:
            x (np.ndarray): The input to the activation function
        Returns:
            np.ndarray: The output of the activation function
        """
        return np.maximum(0, x)

    def _drelu(self, x: np.ndarray) -> np.ndarray:
        """
        The derivative of the ReLU activation function
        Args:
            x (np.ndarray): The input to the activation function
        Returns:
            np.ndarray: The output of the derivative of the activation function
        """
        return x > 0


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        """
        Args:
            input_nodes (int): The number of input nodes
            hidden_nodes (int): The number of hidden nodes
            output_nodes (int): The number of output nodes
        """
        self.input_hidden = Layer(input_nodes, hidden_nodes, "sigmoid")
        self.hidden_output = Layer(hidden_nodes, output_nodes, "sigmoid")

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Overload the call operator to make the neural network callable
        Args:
            input (np.ndarray): The input to the neural network
        Returns:
            np.ndarray: The output of the neural network
        """
        return self._forward(input)

    def _forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network
        Args:
            input (np.ndarray): The input to the neural network
        Returns:
            np.ndarray: The output of the neural network
        """
        self.input = input
        hidden = self.input_hidden(input)
        output = self.hidden_output(hidden)
        return output

    def train(
        self,
        learning_rate: float,
        epochs: int,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        learning_rate_decay: float = 1,
        lr_decay_epoch: int = 1,
    ) -> None:
        """
        Train the neural network
        Args:
            learning_rate (float): The learning rate to use for training
            epochs (int): The number of epochs to train the neural network
            images (np.ndarray): The images to train on
            labels (np.ndarray): The labels of the images
            learning_rate_decay (float, optional): The learning rate decay factor. Defaults to 1.
            lr_decay_epoch (int, optional): The number of epochs before the learning rate decays. Defaults to 1.
        """
        num_correct = 0
        for epoch in range(epochs):
            for image, label in zip(train_images, train_labels):
                image.shape += (1,)
                label.shape += (1,)

                # Forward propagation
                output = self._forward(image)

                # Calculate error
                error = 1 / len(output) * np.sum((output - label) ** 2, axis=0)
                num_correct += int(np.argmax(output) == np.argmax(label))

                # Backpropagation
                delta_output = output - label
                delta_hidden = self.hidden_output.backward(delta_output, learning_rate)
                self.input_hidden.backward(delta_hidden, learning_rate)

            print(
                f"Epoch {epoch + 1} - Train accuracy: {num_correct / len(train_images) * 100:.2f}%, Loss: {error[0]}"
            )
            print(f"Test accuracy: {self.accuracy(test_images, test_labels) * 100:.2f}%")
            num_correct = 0

            print(f"Learning rate: {learning_rate}")
            if epoch % lr_decay_epoch == 0 and epoch != 0:
                learning_rate *= learning_rate_decay

    def accuracy(self, images: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the accuracy of the neural network
        Args:
            images (np.ndarray): The images to calculate the accuracy on
            labels (np.ndarray): The labels of the images
        Returns:
            float: The accuracy of the neural network
        """
        num_correct = 0
        for image, label in zip(images, labels):
            image.shape += (1,)
            label.shape += (1,)
            output = self._forward(image)
            num_correct += int(np.argmax(output) == np.argmax(label))
        return num_correct / len(images)

    def predict(self, image: np.ndarray) -> int:
        """
        Predict the label of an image
        Args:
            image (np.ndarray): The image to predict
        Returns:
            int: The predicted label
        """
        image.shape += (1,)
        output = self._forward(image)
        return np.argmax(output)


def get_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument(
        "-e",
        "--epochs",
        action=EpochAction,
        type=int,
        default=3,
        help="Number of epochs to train the neural network",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        action=LearningRateAction,
        type=float,
        default=0.01,
        help="Learning rate for the neural network",
    )
    parser.add_argument(
        "-ld",
        "--learning-rate-decay",
        action=LearningRateAction,
        type=float,
        default=1,
        help="Learning rate decay for the neural network",
    )
    parser.add_argument(
        "-le",
        "--lr-decay-epoch",
        action=EpochAction,
        type=int,
        default=1,
        help="Epochs before the learning rate decays",
    )
    parser.add_argument(
        "-c",
        "--custom-dataset",
        action="store_true",
        help="Use a custom dataset instead of the MNIST dataset",
    )
    parser.add_argument(
        "-d",
        "--draw",
        action="store_true",
        help="Draw a number to predict",
    )

    args = parser.parse_args()
    return (
        args.epochs,
        args.learning_rate,
        args.learning_rate_decay,
        args.lr_decay_epoch,
        args.custom_dataset,
        args.draw,
    )


def main():
    epochs, learning_rate, learning_rate_decay, lr_decay_epoch, custom_dataset, draw = (
        get_arguments()
    )
    nn = NeuralNetwork(784, 50, 10)
    if custom_dataset:
        train_images, train_labels, test_images, test_labels = DataLoader().load_custom_data()
    else:
        train_images, train_labels, test_images, test_labels = DataLoader().load_mnist_data()
    nn.train(
        learning_rate,
        epochs,
        train_images,
        train_labels,
        test_images,
        test_labels,
        learning_rate_decay,
        lr_decay_epoch,
    )
    if draw and not custom_dataset:
        drawing_app = DrawingApp()
        print("Press 'R' to clear the window, 'ENTER' to continue and 'ESC' to exit")
        while drawing_app.running:
            image = drawing_app.get_image()
            if image is not None:
                plt.imshow(image.reshape(28, 28), cmap="gray")
                plt.title(f"Prediction: {nn.predict(image)}")
                plt.show()
    else:
        while True:
            index = int(input(f"Enter an index[0 - {len(test_images) - 1}](-1 to exit): "))
            if index == -1:
                break
            image = test_images[index]
            plt.imshow(image.reshape(28, 28), cmap="gray")
            plt.title(f"Prediction: {nn.predict(image)}")
            plt.show()


if __name__ == "__main__":
    main()
