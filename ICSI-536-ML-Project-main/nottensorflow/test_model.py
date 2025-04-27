import numpy as np
from nottensorflow.neural_net import Model
from nottensorflow.Layer import Dense
from nottensorflow.activation_fns import ReLU, Softmax
from nottensorflow.loss_functions import CrossEntropyLoss
from nottensorflow.image_processing import process_image
import os
import glob

def load_images(directory):
    """Load and preprocess images from a directory."""
    images = []
    labels = []
    
    for img_path in glob.glob(os.path.join(directory, "*.jpg")):
        label = 1 if "Fracture" in img_path else 0
        
        img = process_image(img_path)
        images.append(img.flatten())
        labels.append(label)
    
    return np.array(images), np.array(labels)

def main():
    # Load training data
    X_train, y_train = load_images("data/input")
    
    # Convert labels to one-hot encoding
    num_classes = 2
    y_train_one_hot = np.zeros((y_train.size, num_classes))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1
    
    # Create model
    input_size = X_train.shape[1]
    model = Model()
    model.add(Dense(input_size, 128))
    model.add(ReLU())
    model.add(Dense(128, 64))
    model.add(ReLU())
    model.add(Dense(64, num_classes))
    model.add(Softmax())
    
    # Train model
    loss_fn = CrossEntropyLoss()
    model.train(X_train, y_train_one_hot, epochs=100, learning_rate=0.01, loss_fn=loss_fn)
    
    # Test model on training data
    predictions = model.predict(X_train)
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_labels == y_train)
    print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")
    
    # Print predictions for each image
    for i, (pred, true) in enumerate(zip(pred_labels, y_train)):
        print(f"Image {i+1}: Predicted {'Fracture' if pred == 1 else 'No Fracture'}, "
              f"True {'Fracture' if true == 1 else 'No Fracture'}")

if __name__ == "__main__":
    main() 