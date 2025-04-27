from nottensorflow.Layer import Layer
import numpy as np

# ReLU activation
class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate):
        return output_grad * (self.input > 0)

class Softmax(Layer):
    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))  # for numerical stability
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_grad, learning_rate):
        # Assume output_grad is dL/dY (loss gradient w.r.t. softmax output)
        # This is a simplified version assuming the use of cross-entropy loss directly
        return output_grad

class Sigmoid(Layer):

    def forward(self, input):
        """
        Turns the elements of `z` into probabilities by plugging them into logistic function.
        """
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_grad, learning_rate):
        """
        First derivative of the sigmoid/logistic function
        """
        return output_grad * self.output * (1 - self.output)
    