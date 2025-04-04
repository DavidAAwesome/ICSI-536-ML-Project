import numpy as np
from abc import abstractmethod, ABC

class ActivationFunc(ABC):
    @abstractmethod
    def activate(z):
        pass
    @abstractmethod
    def deriv(z):
        pass

class Softmax(ActivationFunc):
    def __init__(self):
        pass

    @classmethod
    def activate(cls, z: np.ndarray) -> np.ndarray:
        """
        `z`: numpy arrray

        Turns the elements of `z` into 'probabilities' by scaling `z` by 1/sum(`z`'s elements)
        """
        ez = np.exp(z) 
        return ez / np.sum(ez)
    
    @classmethod
    def deriv(cls, z: np.ndarray) -> np.ndarray:
        """
        `z`: numpy arrray 
        
        Applies the Jacobian of the softmax function to the vector `z`.
        """
        z = Softmax.activate(z)
        z = z.reshape((1,-1))
        return np.diagflat(z) - z.T @ z

class ReLU(ActivationFunc):
    
    def __init__(self):
        pass

    @classmethod
    def activate(cls, z):
        """
        `z`: numpy arrray

        ReLU(`z`) = max{0, `z`}. Gets rid of all negative entries of `z`
        """
        return np.maximum(0, z)
    
    @classmethod
    def deriv(cls, z):
        """
        `z`: numpy arrray

        Returns 0 if entry of `z` is negative, else 1.
        """
        return (z >= 0).astype(float)
    
class Sigmoid(ActivationFunc):

    def __init__(self):
        pass
    
    @classmethod
    def activate(cls, z):
        """
        `z`: numpy arrray

        Turns the elements of `z` into probabilities by plugging them into logistic function.
        """
        return 1 / (1 + np.exp(-z))
    
    @classmethod
    def deriv(cls, z):
        """
        `z`: numpy arrray

        First derivative of the sigmoid/logistic function
        """
        return np.exp(-z) / ((1 + np.exp(-z))**2)