import numpy as np
import matplotlib.pyplot as plt
from utils import DataLoader, EpochAction, LearningRateAction, DrawingApp
import argparse


class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = "sigmoid"):
        """
        Args:
            input_size (int): The number of input nodes
            output_size (int): The number of output nodes
            activation (str, optional): The activation function to use. Defaults to "sigmoid".
        """
        # initialize the weights using he-initialization
        self.weights: np.ndarray = np.random.randn(input_size, output_size) * np.sqrt(
            2 / input_size
        )
        self.biases: np.ndarray = np.zeros((1, output_size))
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

        pre_activation_output: np.ndarray = self.input @ self.weights + self.biases

        # Apply activation functions
        if self.activation == "sigmoid":
            self.output: np.ndarray = self._sigmoid(pre_activation_output)
        elif self.activation == "relu":
            self.output: np.ndarray = self._relu(pre_activation_output)
        elif self.activation == "softmax":
            self.output: np.ndarray = self._softmax(pre_activation_output)

        else:
            print(f"The activation function entered does not exist!\nUse either relu or softmax")

        return self.output

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
        elif self.activation == "softmax":
            delta: np.ndarray = self._dsoftmax(self.output, gradient)

        # calculate the gradients with respect to the weights and bias
        weights_gradient: np.ndarray = self.input.T @ delta
        biases_gradient: np.ndarray = np.sum(delta, axis=0, keepdims=True)

        # clip the gradients to prevent exploding gradients
        weights_gradient = np.clip(weights_gradient, -1.0, 1.0)
        biases_gradient = np.clip(biases_gradient, -1.0, 1.0)

        # update the weights and bias
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return delta @ self.weights.T

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

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        The softmax activation function
        Args:
            x (np.ndarray): The input to the activation function
        Returns:
            np.ndarray: The output of the activation function
        """
        exponential_values: np.ndarray = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exponential_values / np.sum(exponential_values, axis=-1, keepdims=True)

    def _dsoftmax(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        The derivative of the softmax activation function
        Args:
            x (np.ndarray): The input to the activation function
            gradient (np.ndarray): The gradient of the loss with respect to the output of the layer
        Returns:
            np.ndarray: The output of the derivative of the activation function
        """
        for i, d_value in enumerate(gradient):
            if len(d_value.shape) == 1:
                d_value = d_value.reshape(-1, 1)
            jacobian_matrix = np.diagflat(d_value) - (d_value @ d_value.T)
            gradient[i] = jacobian_matrix @ self.output[i]

        return gradient


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        """
        Args:
            input_nodes (int): The number of input nodes
            hidden_nodes (int): The number of hidden nodes
            output_nodes (int): The number of output nodes
        """
        self.input_hidden = Layer(input_nodes, hidden_nodes, "relu")
        self.hidden_output = Layer(hidden_nodes, output_nodes, "softmax")

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
        for epoch in range(epochs):
            output: np.ndarray = self._forward(train_images)

            # calculate the loss using the cross-entropy loss function
            epsilon = 1e-10
            error: float = -np.mean(train_labels * np.log(output + epsilon))

            # calculate the accuracy
            predicted_labels: np.ndarray = np.argmax(output, axis=1)
            true_labels: np.ndarray = np.argmax(train_labels, axis=1)
            accuracy: float = np.mean(predicted_labels == true_labels)

            # calculate the gradient of the loss with respect to the output
            delta_output = (output - train_labels) / output.shape[0]

            # backpropagate the gradient
            delta_hidden = self.hidden_output.backward(delta_output, learning_rate)
            delta_input = self.input_hidden.backward(delta_hidden, learning_rate)

            print(
                f"Epoch {epoch + 1} - Train accuracy: {accuracy * 100:.2f}% - Loss: {error:.4f}",
                end=" - ",
            )
            print(f"Test accuracy: {self.accuracy(test_images, test_labels) * 100:.2f}%")

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
        output: np.ndarray = self._forward(images)
        predicted_labels: np.ndarray = np.argmax(output, axis=1)
        true_labels: np.ndarray = np.argmax(labels, axis=1)
        accuracy: float = np.mean(predicted_labels == true_labels)
        return accuracy

    def predict(self, image: np.ndarray) -> int:
        """
        Predict the label of an image
        Args:
            image (np.ndarray): The image to predict
        Returns:
            int: The predicted label
        """
        output: np.ndarray = self._forward(image)
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
