# Neural Network From Scratch

This Python script implements a simple neural network for image classification. The neural network is designed with an input layer of 784 nodes (assuming 28x28 pixel images), a hidden layer with 20 nodes, and an output layer with 10 nodes (for classification). The training is performed using backpropagation with stochastic gradient descent.

## Usage

To use the script, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/b14ucky/nn-from-scratch.git
    cd nn-from-scratch
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the script with the desired parameters:

    ```bash
    python neural_network.py -e <epochs> -l <learning_rate> -ld <learning_rate_decay> -le <lr_decay_epoch> -c -d
    ```

    - **-e, --epochs**: Number of epochs to train the neural network (default is 3).
    - **-l, --learning-rate**: Learning rate for the neural network (default is 0.01).
    - **-ld, --learning-rate-decay**: Learning rate decay factor (default is 1).
    - **-le, --lr-decay-epoch**: Epochs before the learning rate decays (default is 1).
    - **-c, --custom-dataset**: Use a custom dataset instead of the MNIST dataset.
    -  **-d, --draw**: Draw a number to predict.

## Code Structure

- **neural_network.py**: Main script for training and testing the neural network.
- **utils.py**: Contains utility functions for loading data.

## Neural Network Class

- **NeuralNetwork**: The class implementing the neural network with methods for training (`train`) and prediction (`predict`).

## Training Process

The training process involves iterating through the specified number of epochs, performing forward and backward propagation for each input image, and updating the weights and biases of the neural network accordingly. The learning rate can decay over epochs to fine-tune the training process.

## Drawing and Prediction

If the `-d` or `--draw` option is specified and a custom dataset is not used, the script will open a drawing application. You can draw a number, and the trained neural network will predict the drawn digit. This feature is useful for testing the trained neural network on new data. You can also test the neural network on individual images from the dataset by not specifying the `-d` option.

## Command-line Arguments

- **-e, --epochs**: Number of training epochs.
- **-l, --learning-rate**: Initial learning rate.
- **-ld, --learning-rate-decay**: Learning rate decay factor.
- **-le, --lr-decay-epoch**: Epochs before the learning rate decays.
- **-c, --custom-dataset**: Flag to use a custom dataset.
- **-d, --draw**: Flag to draw a number for prediction.

## Dataset

The script supports training on either the default MNIST dataset or a custom dataset (specified using the `-c` flag). This custom dataset was created and used to train my number recognition model, which is employed in the [sudokuAI project](https://github.com/b14ucky/sudokuAI).

Feel free to explore and modify the script for your specific use case or dataset.
