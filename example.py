import time
from nottensorflow.image_processing import process_image
from nottensorflow.neural_net import Model, Dense, MeanSquareLoss
from nottensorflow.activation_fns import ReLU, Softmax, Sigmoid
import os
import numpy as np
from nottensorflow.cross_validation import cross_validation

NUM_CLASSES = 7

## Read in data from CSV
base_dir = os.path.dirname(__file__)
bone_yolo_dir = os.path.join(base_dir, 'data', 'BoneFractureYolo8')
train_path = os.path.join(bone_yolo_dir, 'train_extracted.csv')
train_data = np.loadtxt(train_path, dtype=float, delimiter=',')

X_train = train_data[:-1]
y_train = train_data[-1]


## Init and Train model
start_time = time.time()

img_size = 224 * 224
my_model = (Model(SGD=True)
            .add(Dense(img_size, 16))
            .add(ReLU())
            .add(Dense(16, NUM_CLASSES))
            .add(Softmax()))
# my_model.train(x=X_train, y=y_train, epochs=200, learning_rate=0.1, loss_fn=MeanSquareLoss())

models = cross_validation(my_model.layers, X_train, y_train, epochs=200, learning_rate=0.1, loss_fn=MeanSquareLoss(), passes=5)
for i in range(len(models)):
    pass


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Time: {elapsed_time}")

## Test model

### Get random image
n = 43 # "Random"


test_images_path = os.path.join(bone_yolo_dir, 'test', 'images')
random_image_path = os.listdir(test_images_path)[n]
random_image_path = os.path.join(test_images_path, random_image_path)
print(f'Label: {random_image_path}')
random_image = process_image(random_image_path)

### Get corresponding label
test_labels_path = os.path.join(bone_yolo_dir, 'test', 'labels')
random_label_path = os.listdir(test_labels_path)[n]
random_label_path = os.path.join(test_labels_path, random_label_path)
print(f'Label: {random_label_path}')

with open(random_label_path) as file:
    contents = file.read()
    true_label = int(contents[0]) if contents else -1 # File empty == No fracture
predict_label = np.argmax(my_model.predict(random_image))

print(f"Prediction: {predict_label}, Truth: {true_label}")