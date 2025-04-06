import numpy as np
from neural_net import Layer

class Softmax(Layer):
  
    def activate(self, z: np.ndarray) -> np.ndarray:
        """
        Turns the elements of `z` into 'probabilities' by scaling `z` by 1/sum(`z`'s elements)
        """
        ez = np.exp(z) 
        return ez / np.sum(ez)
    
    def deriv(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the Jacobian of the softmax function to the vector `z`.
        """
        z = self.activate(z)
        z = z.reshape((1,-1))
        return np.diagflat(z) - z.T @ z

class ReLU(Layer):
    
    def activate(self, z):
        """  
        ReLU(`z`) = max{0, `z`}. Gets rid of all negative entries of `z`
        """
        return np.maximum(0, z)
    
    def deriv(self, z):
        """
        Returns 0 if entry of `z` is negative, else 1.
        """
        return (z >= 0).astype(float)
    
class Sigmoid(Layer):
    
    def activate(self, z):
        """
        Turns the elements of `z` into probabilities by plugging them into logistic function.
        """
        return 1 / (1 + np.exp(-z))
    
    def deriv(self, z):
        """
        First derivative of the sigmoid/logistic function
        """
        return np.exp(-z) / ((1 + np.exp(-z))**2)