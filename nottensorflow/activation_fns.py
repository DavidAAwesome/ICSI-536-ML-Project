import numpy as np
from neural_net import Layer

class Softmax(Layer):
  
    def forward(self, input):
        """
        Turns the elements of `z` into 'probabilities' by scaling `z` by 1/sum(`z`'s elements)
        """
        ez = np.exp(input) 
        return ez / np.sum(ez)
    
    def backward(self, output_grad, learning_rate) -> np.ndarray:
        """
        Applies the Jacobian of the softmax function to the vector `z`.
        """
        z = self.forward(z)
        z = z.reshape((1,-1))
        return np.diagflat(z) - z.T @ z

class ReLU(Layer):
    
    def forward(self, input):
        """  
        ReLU(`z`) = max{0, `z`}. Gets rid of all negative entries of `z`
        """
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output_grad, learning_rate):
        """
        Returns 0 if entry of `z` is negative, else 1.
        """
        return (self.input >= 0).astype(float)
    
class Sigmoid(Layer):
    
    def forward(self, z):
        """
        Turns the elements of `z` into probabilities by plugging them into logistic function.
        """
        return 1 / (1 + np.exp(-z))
    
    def deriv(self, z):
        """
        First derivative of the sigmoid/logistic function
        """
        return np.exp(-z) / ((1 + np.exp(-z))**2)