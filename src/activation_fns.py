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
        z: n-dimensional vector

        Turns the elements of `z` into 'probabilities' by scaling `z` by 1/sum(`z` elements)
        """
        ez = np.exp(z) 
        return ez / np.sum(ez)
    
    @classmethod
    def deriv(cls, z: np.ndarray) -> np.ndarray:
        """
        z: n-dimensional vector
        
        Applies the Jacobian of the softmax function to the vector `z`.
        """
        assert z.ndim == 1
        return np.array([
            [-z[i] * z[j] if i != j else z[i]*(1-z[j]) 
             for i in range(z.size)] 
            for j in range(z.size)
        ])