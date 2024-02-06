import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist_data, load_my_data


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.w_input_hidden = np.random.uniform(-0.5, 0.5, (hidden_nodes, input_nodes))
        self.w_hidden_output = np.random.uniform(-0.5, 0.5, (output_nodes, hidden_nodes))
        self.b_input_hidden = np.zeros((hidden_nodes, 1))
        self.b_hidden_output = np.zeros((output_nodes, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(
        self,
        learning_rate,
        epochs,
        images,
        labels,
        learning_rate_decay=1,
        lr_decay_epoch=1,
    ):
        num_correct = 0
        for epoch in range(epochs):
            for image, label in zip(images, labels):
                image.shape += (1,)
                label.shape += (1,)

                # Forward propagation (input to hidden)
                hidden_pre = self.b_input_hidden + self.w_input_hidden @ image
                hidden = self._sigmoid(hidden_pre)

                # Forward propagation (hidden to output)
                output_pre = self.b_hidden_output + self.w_hidden_output @ hidden
                output = self._sigmoid(output_pre)

                # Calculate error
                error = 1 / len(output) * np.sum((output - label) ** 2, axis=0)
                num_correct += int(np.argmax(output) == np.argmax(label))

                # Backpropagation (output to hidden)
                delta_output = output - label
                self.w_hidden_output += -learning_rate * delta_output @ np.transpose(hidden)
                self.b_hidden_output += -learning_rate * delta_output

                # Backpropagation (hidden to input)
                delta_hidden = (
                    np.transpose(self.w_hidden_output) @ delta_output * (hidden * (1 - hidden))
                )
                self.w_input_hidden += -learning_rate * delta_hidden @ np.transpose(image)
                self.b_input_hidden += -learning_rate * delta_hidden

            print(f"Epoch {epoch + 1}: {num_correct / len(images) * 100:.2f}%")
            num_correct = 0

            print(f"Learning rate: {learning_rate}")
            if epoch % lr_decay_epoch == 0:
                learning_rate *= learning_rate_decay

    def predict(self, image):
        image.shape += (1,)

        # Forward propagation (input to hidden)
        hidden_pre = self.b_input_hidden + self.w_input_hidden @ image
        hidden = self._sigmoid(hidden_pre)

        # Forward propagation (hidden to output)
        output_pre = self.b_hidden_output + self.w_hidden_output @ hidden
        output = self._sigmoid(output_pre)

        return np.argmax(output)


if __name__ == "__main__":
    nn = NeuralNetwork(784, 20, 10)
    images, labels = load_mnist_data()
    nn.train(0.01, 3, images, labels)
    while True:
        index = int(input(f"Enter an index[0 - {len(images) - 1}]: "))
        if index == -1:
            break
        image = images[index]
        plt.imshow(image.reshape(28, 28), cmap="gray")
        plt.title(f"Prediction: {nn.predict(image)}")
        plt.show()
