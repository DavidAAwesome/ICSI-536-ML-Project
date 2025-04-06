import time
from nottensorflow.image_processing import process_image
from nottensorflow.neural_net import Model, Dense, ReLU, MeanSquareLoss
import os
import numpy as np

num_classes = 7

## Helper functions
def extract_class_label_from_file(file_path: str):
    """
    Extracts class ID (first number) from YOLO dataset label. Returns as
    numpy array.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    nums = content.split()
    if not nums:
        return None
    class_id = int(nums[0])
    return class_id

def onehot_encode(y: int, num_classes: int):
    onehot_y = np.zeros(num_classes)
    onehot_y[y] = 1
    return onehot_y

## Load images & labels as numpy arrays
base_dir = os.path.dirname(__file__)
bone_yolo_dir = os.path.join(base_dir, 'data', 'BoneFractureYolo8')
images_dir = os.path.join(bone_yolo_dir, 'train', 'images')
labels_dir = os.path.join(bone_yolo_dir, 'train', 'labels')

limit = 1000 # Only scan in `limit` number of images
X_rows = [] 
y_rows = []
for label_entry in os.scandir(labels_dir):

    if len(y_rows) >= limit:
        break
    # Accumate labels in list, to be converted to numpy array later
    y = extract_class_label_from_file(label_entry.path)
    if y is None: # Drop non-fractured images
        continue
    onehot_y = onehot_encode(y, num_classes)
    y_rows.append(onehot_y)

    # Do same for images
    image_name = label_entry.name.replace('.txt', '.jpg')
    image_path = os.path.join(images_dir, image_name)
    X_rows.append(process_image(image_path))

X_train = np.array(X_rows)
y_train = np.array(y_rows)

## Init and Train model
start_time = time.time()

img_size = 224 * 224
my_model = (Model()
            .add(Dense(img_size, 16))
            .add(ReLU())
            .add(Dense(16, num_classes)))

my_model.train(x=X_train, y=y_train, epochs=200, learning_rate=0.1, loss_fn=MeanSquareLoss())

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Time: {elapsed_time}")


# TODO: Cross validation

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