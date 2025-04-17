import numpy as np
import os
from nottensorflow.image_processing import process_image

NUM_CLASSES = 7

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

def dummy_code(y: int, num_classes: int):
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
    onehot_y = dummy_code(y, NUM_CLASSES)
    y_rows.append(onehot_y)

    # Do same for images
    image_name = label_entry.name.replace('.txt', '.jpg')
    image_path = os.path.join(images_dir, image_name)
    X_rows.append(process_image(image_path))

X_train = np.array(X_rows)
y_train = np.array(y_rows)

train_data = np.concatenate([X_train, y_train], axis=1)
np.savetxt("data/BoneFractureYolo8/train_extracted.csv", train_data, delimiter=",", fmt='%.3f')