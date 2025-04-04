import numpy as np



class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad, learning_rate):
        raise NotImplementedError



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



class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate):
        return output_grad * (self.input > 0)


class MSELoss:
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target) ** 2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.target.size



class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def predict(self, x):
        return self.forward(x)

    def train(self, x, y, epochs, learning_rate, loss_fn):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(x)

            # Compute loss
            loss = loss_fn.forward(output, y)

            # Backward pass
            grad = loss_fn.backward()
            self.backward(grad, learning_rate)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
