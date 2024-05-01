import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from utils import (
    DataLoader,
    NonpositiveIntAction,
    LearningRateAction,
    DrawingApp,
    ImageAugmentation,
    plot,
)
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
        self.weights: NDArray = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases: NDArray = np.zeros((1, output_size))
        self.activation: str = activation

        # adam optimizer parameters
        self.m_weights: NDArray = np.zeros_like(self.weights)
        self.v_weights: NDArray = np.zeros_like(self.weights)
        self.m_biases: NDArray = np.zeros_like(self.biases)
        self.v_biases: NDArray = np.zeros_like(self.biases)
        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.epsilon: float = 1e-8

    def __call__(self, input: NDArray) -> NDArray:
        """
        Overload the call operator to make the layer callable
        Args:
            input (NDArray): The input to the layer
        Returns:
            NDArray: The output of the layer
        """
        return self._forward(input)

    def _forward(self, input: NDArray) -> NDArray:
        """
        Forward pass through the layer
        Args:
            input (NDArray): The input to the layer
        Returns:
            NDArray: The output of the layer
        """
        self.input: NDArray = input

        pre_activation_output: NDArray = self.input @ self.weights + self.biases

        # Apply activation functions
        if self.activation == "sigmoid":
            self.output: NDArray = self._sigmoid(pre_activation_output)
        elif self.activation == "relu":
            self.output: NDArray = self._relu(pre_activation_output)
        elif self.activation == "softmax":
            self.output: NDArray = self._softmax(pre_activation_output)

        else:
            print(f"The activation function entered does not exist!\nUse either relu or softmax")

        return self.output

    def backward(self, gradient: NDArray, learning_rate: float, t: int) -> NDArray:
        """
        Backward pass through the layer
        Args:
            gradient (NDArray): The gradient of the loss with respect to the output of the layer
            learning_rate (float): The learning rate to use for updating the weights and bias
            t (int): The current iteration for the Adam optimizer
        Returns:
            NDArray: The gradient of the loss with respect to the input of the layer
        """
        # calculate the delta l
        if self.activation == "sigmoid":
            delta: NDArray = gradient * self._dsigmoid(self.output)
        elif self.activation == "relu":
            delta: NDArray = gradient * self._drelu(self.output)
        elif self.activation == "softmax":
            delta: NDArray = self._dsoftmax(self.output, gradient)

        # calculate the gradients with respect to the weights and bias
        weights_gradient: NDArray = self.input.T @ delta
        biases_gradient: NDArray = np.sum(delta, axis=0, keepdims=True)

        # clip the gradients to prevent exploding gradients
        weights_gradient = np.clip(weights_gradient, -1.0, 1.0)
        biases_gradient = np.clip(biases_gradient, -1.0, 1.0)

        # update the weights and bias
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        # calculate the moving average of the gradients at time step t
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * weights_gradient
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * weights_gradient**2
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * biases_gradient
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * biases_gradient**2

        # calculate the corrected moving averages
        corrected_m_weights = self.m_weights / (1 - self.beta1**t)
        corrected_v_weights = self.v_weights / (1 - self.beta2**t)
        corrected_m_biases = self.m_biases / (1 - self.beta1**t)
        corrected_v_biases = self.v_biases / (1 - self.beta2**t)

        # update the weights and bias using the Adam optimizer
        self.weights -= (
            learning_rate * corrected_m_weights / (np.sqrt(corrected_v_weights) + self.epsilon)
        )
        self.biases -= (
            learning_rate * corrected_m_biases / (np.sqrt(corrected_v_biases) + self.epsilon)
        )

        return delta @ self.weights.T

    def _sigmoid(self, x: NDArray) -> NDArray:
        """
        The sigmoid activation function
        Args:
            x (NDArray): The input to the activation function
        Returns:
            NDArray: The output of the activation function
        """
        return 1 / (1 + np.exp(-x))

    def _dsigmoid(self, x: NDArray) -> NDArray:
        """
        The derivative of the sigmoid activation function
        Args:
            x (NDArray): The input to the activation function
        Returns:
            NDArray: The output of the derivative of the activation function
        """
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _relu(self, x: NDArray) -> NDArray:
        """
        The ReLU activation function
        Args:
            x (NDArray): The input to the activation function
        Returns:
            NDArray: The output of the activation function
        """
        return np.maximum(0, x)

    def _drelu(self, x: NDArray) -> NDArray:
        """
        The derivative of the ReLU activation function
        Args:
            x (NDArray): The input to the activation function
        Returns:
            NDArray: The output of the derivative of the activation function
        """
        return x > 0

    def _softmax(self, x: NDArray) -> NDArray:
        """
        The softmax activation function
        Args:
            x (NDArray): The input to the activation function
        Returns:
            NDArray: The output of the activation function
        """
        exponential_values: NDArray = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exponential_values / np.sum(exponential_values, axis=-1, keepdims=True)

    def _dsoftmax(self, x: NDArray, gradient: NDArray) -> NDArray:
        """
        The derivative of the softmax activation function
        Args:
            x (NDArray): The input to the activation function
            gradient (NDArray): The gradient of the loss with respect to the output of the layer
        Returns:
            NDArray: The output of the derivative of the activation function
        """
        for i, d_value in enumerate(gradient):
            if len(d_value.shape) == 1:
                d_value = d_value.reshape(-1, 1)
            jacobian_matrix = np.diagflat(d_value) - (d_value @ d_value.T)
            gradient[i] = jacobian_matrix @ self.output[i]

        return gradient


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: list[int], output_nodes: int) -> None:
        """
        Args:
            input_nodes (int): The number of input nodes
            hidden_nodes (list[int]): The number of hidden nodes
            output_nodes (int): The number of output nodes
        """
        self.l1 = Layer(input_nodes, hidden_nodes[0], "relu")
        self.l2 = Layer(hidden_nodes[0], hidden_nodes[1], "relu")
        self.l3 = Layer(hidden_nodes[1], output_nodes, "softmax")

    def __call__(self, input: NDArray) -> NDArray:
        """
        Overload the call operator to make the neural network callable
        Args:
            input (NDArray): The input to the neural network
        Returns:
            NDArray: The output of the neural network
        """
        return self._forward(input)

    def _forward(self, input: NDArray) -> NDArray:
        """
        Forward pass through the neural network
        Args:
            input (NDArray): The input to the neural network
        Returns:
            NDArray: The output of the neural network
        """
        output1 = self.l1(input)
        output2 = self.l2(output1)
        output = self.l3(output2)

        return output

    def train(
        self,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        train_images: NDArray,
        train_labels: NDArray,
        test_images: NDArray,
        test_labels: NDArray,
        plot_stats: bool = False,
        learning_rate_decay: float = 0.001,
    ) -> None:
        """
        Train the neural network
        Args:
            learning_rate (float): The learning rate to use for training
            epochs (int): The number of epochs to train the neural network
            batch_size (int): The batch size to use for training
            images (NDArray): The images to train on
            labels (NDArray): The labels of the images
            learning_rate_decay (float, optional): The learning rate decay factor. Defaults to 1.
        """
        loss_plot = []
        train_accuracy_plot = []
        test_accuracy_plot = []
        t = 0
        for epoch in range(epochs):
            for i in range(0, len(train_images), batch_size):
                batch_images = train_images[i : i + batch_size]
                batch_labels = train_labels[i : i + batch_size]

                for i, image in enumerate(batch_images):
                    image = ImageAugmentation(image)()
                    batch_images[i] = image

                output: NDArray = self._forward(batch_images)

                # calculate the gradient of the loss with respect to the output
                delta_output = (output - batch_labels) / output.shape[0]

                # backpropagate the gradient
                t += 1
                delta_hidden2 = self.l3.backward(delta_output, learning_rate, t)
                delta_hidden1 = self.l2.backward(delta_hidden2, learning_rate, t)
                delta_input = self.l1.backward(delta_hidden1, learning_rate, t)

            # decay the learning rate
            learning_rate /= 1 + learning_rate_decay

            # calculate the loss using the cross-entropy loss function
            epsilon = 1e-10
            error: float = -np.mean(batch_labels * np.log(output + epsilon))
            loss_plot.append(error)

            # calculate the accuracy
            predicted_labels: NDArray = np.argmax(output, axis=1)
            true_labels: NDArray = np.argmax(batch_labels, axis=1)
            train_accuracy: float = np.mean(predicted_labels == true_labels) * 100
            train_accuracy_plot.append(train_accuracy)

            print(
                f"Epoch {epoch + 1} - Train accuracy: {train_accuracy:.2f}% - Loss: {error:.4f}",
                end=" - ",
            )

            # calculate the test accuracy
            test_accuracy = self.accuracy(test_images, test_labels) * 100
            test_accuracy_plot.append(test_accuracy)
            print(f"Test accuracy: {test_accuracy:.2f}%")

            # plot the loss and accuracy
            if plot_stats:
                plot(loss_plot, test_accuracy_plot, train_accuracy_plot)

    def accuracy(self, images: NDArray, labels: NDArray) -> float:
        """
        Calculate the accuracy of the neural network
        Args:
            images (NDArray): The images to calculate the accuracy on
            labels (NDArray): The labels of the images
        Returns:
            float: The accuracy of the neural network
        """
        output: NDArray = self._forward(images)
        predicted_labels: NDArray = np.argmax(output, axis=1)
        true_labels: NDArray = np.argmax(labels, axis=1)
        accuracy: float = np.mean(predicted_labels == true_labels)
        return accuracy

    def predict(self, image: NDArray) -> int:
        """
        Predict the label of an image
        Args:
            image (NDArray): The image to predict
        Returns:
            int: The predicted label
        """
        output: NDArray = self._forward(image)
        return np.argmax(output)


def get_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument(
        "-e",
        "--epochs",
        action=NonpositiveIntAction,
        type=int,
        default=3,
        help="Number of epochs to train the neural network",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        action=NonpositiveIntAction,
        type=int,
        default=32,
        help="Batch size for training the neural network",
    )
    parser.add_argument(
        "-p",
        "--plot-stats",
        action="store_true",
        help="Plot the loss and accuracy of the neural network",
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
        default=0.001,
        help="Learning rate decay for the neural network",
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
        args.batch_size,
        args.plot_stats,
        args.learning_rate,
        args.learning_rate_decay,
        args.custom_dataset,
        args.draw,
    )


def main():
    (
        epochs,
        batch_size,
        plot_stats,
        learning_rate,
        learning_rate_decay,
        custom_dataset,
        draw,
    ) = get_arguments()
    nn = NeuralNetwork(784, [512, 512], 10)

    if custom_dataset:
        train_images, train_labels, test_images, test_labels = DataLoader().load_custom_data()
    else:
        train_images, train_labels, test_images, test_labels = DataLoader().load_mnist_data()

    nn.train(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        plot_stats=plot_stats,
        learning_rate_decay=learning_rate_decay,
    )

    if draw and not custom_dataset:
        drawing_app = DrawingApp()
        print("Press 'R' to clear the window, 'ENTER' to continue and 'ESC' to exit")
        while drawing_app.running:
            image = drawing_app.get_image()
            if image is not None:
                plt.clf()
                plt.imshow(image.reshape(28, 28), cmap="gray")
                plt.title(f"Prediction: {nn.predict(image)}")
                plt.show()
    else:
        while True:
            index = int(input(f"Enter an index[0 - {len(test_images) - 1}](-1 to exit): "))
            if index == -1:
                break
            image = test_images[index]
            plt.clf()
            plt.imshow(image.reshape(28, 28), cmap="gray")
            plt.title(f"Prediction: {nn.predict(image)}")
            plt.show()


if __name__ == "__main__":
    main()
