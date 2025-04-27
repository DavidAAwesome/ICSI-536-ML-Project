import numpy as np
from nottensorflow.Layer import Layer

# Mean Squared Error loss
class MeanSquaredLoss:
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target) ** 2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.target.size

class CrossEntropyLoss:
    def forward(self, prediction, target):
        # Add epsilon to avoid log(0)
        self.prediction = prediction
        self.target = target
        epsilon = 1e-15
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        self.loss = -np.sum(target * np.log(prediction)) / prediction.shape[0]
        return self.loss

    def backward(self):
        # Gradient of loss w.r.t. softmax input, assuming softmax was applied before
        return (self.prediction - self.target) / self.target.shape[0]
