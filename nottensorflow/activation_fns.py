import random
import numpy as np
from nottensorflow.neural_net import Layer

class Softmax(Layer):
  
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Turns the elements of `input` into 'probabilities' by scaling `input` by 1/sum(`input`'s elements)
        """
        self.input = input
        # Improve numerical stability
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Simplified gradient for softmax when used with cross-entropy loss
        """
        return output_grad

class ReLU(Layer):
    def forward(self, input):
        """  
        ReLU(`z`) = max{0, `z`}. Gets rid of all negative entries of `z`
        """
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate):
        """
        Returns 0 if entry of `output_grad` is negative, else 1.
        """
        return output_grad * (self.input > 0)
 
class Sigmoid(Layer):
    
    def forward(self, input):
        """
        Turns the elements of `z` into probabilities by plugging them into logistic function.
        """
        self.input = input
        return 1 / (1 + np.exp(-input))
    
    def backward(self, output_grad, learning_rate):
        """
        First derivative of the sigmoid/logistic function
        """
        sig = self.forward(self.input)
        return output_grad * sig * (1 - sig)
    