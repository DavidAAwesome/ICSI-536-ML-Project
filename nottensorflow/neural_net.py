import numpy as np

# Model class
class Model:
    def __init__(self):
        self.layers = []
        self.accuracy_history = []  # track accuracy over epochs
        self.name = "Model"

    def add(self, layer):
        self.layers.append(layer)
        return self

    def set_name(self, name):
        self.name = name
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def predict(self, x):
        return self.forward(x)

    def compute_accuracy(self, x, y):
        predictions = self.predict(x)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(pred_labels == true_labels)

    def train_SGD(self, x, y, epochs, learning_rate, loss_fn, batch_size):
        num_samples = x.shape[0]

        for epoch in range(epochs):
            # Shuffle the data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]

            # Loop over mini-batches
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                x_batch = x[start:end]
                y_batch = y[start:end]

                # Forward pass
                output = self.forward(x_batch)

                # Compute loss
                loss = loss_fn.forward(output, y_batch)

                # Backward pass
                grad = loss_fn.backward()
                self.backward(grad, learning_rate)

            # Compute and store accuracy for this epoch
            epoch_accuracy = self.compute_accuracy(x, y)
            self.accuracy_history.append(epoch_accuracy)

            # Print epoch loss and accuracy
            if (epoch + 1) % 1 == 0:
                full_output = self.forward(x)
                full_loss = loss_fn.forward(full_output, y)
                print(f"Epoch {epoch + 1}, Loss: {full_loss:.6f}, Accuracy: {epoch_accuracy:.4f}")

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
