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

print("Processing images and labels...")
for label_entry in os.scandir(labels_dir):
    if len(y_rows) >= limit:
        break
    
    # Get label
    y = extract_class_label_from_file(label_entry.path)
    if y is None:  # Drop non-fractured images
        continue
    onehot_y = dummy_code(y, NUM_CLASSES)

    # Get corresponding image
    image_name = label_entry.name.replace('.txt', '.jpg')
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found for {label_entry.name}")
        continue
        
    img_features = process_image(image_path)
    if img_features is None:
        print(f"Failed to process image {image_path}")
        continue
        
    X_rows.append(img_features)
    y_rows.append(onehot_y)

print(f"Processed {len(X_rows)} images")

X_train = np.array(X_rows)
y_train = np.array(y_rows)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Combine features and labels horizontally
train_data = np.hstack((X_train, y_train))
print("Combined data shape:", train_data.shape)

# Save to CSV
output_path = os.path.join(bone_yolo_dir, 'train_extracted.csv')
np.savetxt(output_path, train_data, delimiter=",", fmt='%.3f')
print(f"Saved data to {output_path}")