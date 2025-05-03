import tensorflow as tf
from tensorflow.keras.datasets import mnist
import Layer
import loss_functions
import activation_functions
import neural_network
import preformance_metrics
import numpy as np
import Cross_validation

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize images (0-255) → (0-1)
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

# Flatten 28x28 images → vectors of size 784
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
# 10 classes for MNIST
NUM_CLASSES = 10
# converts label in vector e.g ( 5 -> [0, 0, 0, 0, 1,...,0]
def one_hot(y, num_classes=NUM_CLASSES):
    out = np.zeros((y.size, num_classes))
    out[np.arange(y.size), y] = 1
    return out

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# Load data

model = (neural_network.Model()
         .add(Layer.Dense(784, 256))
         .add(activation_functions.ReLU())
         .add(Layer.Dense(256, 512))
         .add(activation_functions.ReLU())
         .add(Layer.Dense(512, 128))
         .add(activation_functions.ReLU())
         .add(Layer.Dense(128, 10))
         .add(activation_functions.Softmax())
         )


loss_fn = loss_functions.CrossEntropyLoss()


model.train_SGD(x_train, y_train_oh, epochs=10, learning_rate=0.1, loss_fn=loss_fn, batch_size=32)

preds = model.predict(x_test)


pred_labels = np.argmax(preds, axis=1)
true_labels = y_test

# Calculate accuracy
con_mat = preformance_metrics.ConfusionMatrix(true_labels, pred_labels, num_classes=NUM_CLASSES)

accuracy = float(con_mat.accuracy())
print(f"Test Accuracy: {accuracy * 100:.2f}%")

models = Cross_validation.cross_validation(model.layers, x_train, y_train_oh, 10, .1, loss_fn=loss_fn,passes=10)
print(models)
