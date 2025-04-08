import numpy as np
from abc import abstractmethod, ABC
from activation_fns import ReLU, Softmax, Sigmoid
import random


class Layer(ABC):
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad, learning_rate, SGD):
        raise NotImplementedError



class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_grad, learning_rate, SGD):
        if SGD:
            i = np.random.choice(self.input.shape[0], 1)
            input = self.input[i]
        else: 
            input = self.input
        weights_grad = np.dot(input.T, output_grad)
        input_grad = np.dot(output_grad, self.weights.T)

        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * np.mean(output_grad, axis=0, keepdims=True)

        return input_grad
    
    
    


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate, SGD):
        if SGD:
            i = np.random.choice(self.input.shape[0], 1)
            input = self.input[i]
        else: 
            input = self.input
        
        return output_grad * (input > 0)



class MeanSquareLoss:
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target) ** 2)

    def backward(self, SGD):
        
        if SGD:
            i = np.random.choice(self.input.shape[0], 1)
            prediction = self.prediction[i]
            target = self.target[i]
        else:
            prediction = self.prediction
            target = self.target
        return 2 * (prediction - target) / target.size



class Model:
    def __init__(self,SGD):
        self.layers = []
        self.SGD = SGD

    def add(self, layer: Layer):
        self.layers.append(layer)
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate, self.SGD)

    def predict(self, x):
        """
        Returns a column vector of natural numbers, representing class IDs.
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def train(self, x, y, epochs, learning_rate, loss_fn):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(x)

            # Compute loss
            loss = loss_fn.forward(output, y)

            # Backward pass
            grad = loss_fn.backward(self.SGD)
            self.backward(grad, learning_rate)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
        
        weights = []
        biases = []
        for layer in self.layers:
            weights.append(layer.weights)
            biases.append(layer.biases)
        return weights, biases

    
    
