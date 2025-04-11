import random
import numpy as np
from neural_net import Layer



class Softmax(Layer):
  
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Turns the elements of `input` into 'probabilities' by scaling `input` by 1/sum(`input`'s elements)
        """
        self.input = input
        # Improve numerical stability
        input -= input.max()
        log_softmax =  input - np.log(input) 
        return np.exp(log_softmax)
    
    def backward(self, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Applies the Jacobian of the softmax function to the vector `output_grad`.
        """
        z = self.forward(self.input).reshape((1,-1)) # 1 x In
        J_z = np.diagflat(z) - np.dot(z.T, z) # In x In
        return np.dot(J_z, output_grad.T) # In x Out 



class ReLU(Layer):
    def forward(self, input):
        """  
        ReLU(`z`) = max{0, `z`}. Gets rid of all negative entries of `z`
        """
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate, SGD):
        """
        Returns 0 if entry of `output_grad` is negative, else 1.
        """
        if SGD:
            i = random.randint(1, self.input.shape[0])
            input = self.input[i]
        else: 
            input = self.input
        
        return output_grad * (input > 0)


 
class Sigmoid(Layer):
    
    def forward(self, input):
        """
        Turns the elements of `z` into probabilities by plugging them into logistic function.
        """
        self.input = input
        return 1 / (1 + np.exp(-input))
    
    def deriv(self, z):
        """
        First derivative of the sigmoid/logistic function
        """
        return np.exp(-z) / ((1 + np.exp(-z))**2)