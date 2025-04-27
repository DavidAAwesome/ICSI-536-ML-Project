import numpy as np
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad, learning_rate):
        raise NotImplementedError

# Fully connected layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_grad, learning_rate):
        weights_grad = np.dot(self.input.T, output_grad)
        input_grad = np.dot(output_grad, self.weights.T)

        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * np.mean(output_grad, axis=0, keepdims=True)

        return input_grad
