import os
import numpy as np
from nottensorflow.image_processing import process_image, augment_and_save

## Helper functions
def extract_class_label_from_file(file_path: str):
    """
    Extracts class ID (first number) from dataset label. Returns as
    numpy array.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    nums = content.split()
    if not nums:
        return None
    class_id = int(nums[0])
    return class_id

def extract(labels_dir, images_dir, limit=-1):
    """
    Extract images and label classes into NumPy array

    Args:
        `labels_dir`: Path to labels
        `images_dir`: Path to images
        `limit`: Number of images to scan in. Pass negative limit to scan all
    """
    X_aug_rows = []
    X_rows = []
    y_rows = []
    num_imgs = 0
    num_blank = 0
    
    for label_entry in os.scandir(labels_dir):
        # Scan in `limit` images only
        if limit >= 0 and num_imgs >= limit: 
            break
    
        # Accumate labels in list, to be converted to numpy array later
        y = extract_class_label_from_file(label_entry.path)
        if y is None: # Drop non-fractured images
            num_blank += 1
            continue
            
        y_rows += [y]
    
        # Do same for images
        image_name = label_entry.name.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_name)
        img = process_image(image_path)
        X_rows += [img.flatten()]
        # Augment images
        aug_img, _ = augment_and_save(img, output_dir='', base_name='', save=False) # Only augment, no save
        X_aug_rows += [aug_img.ravel()]
        
        # Log progress
        num_imgs += 1
        if num_imgs % 200 == 0:
            print(f'Extracted: {num_imgs}')
    
    # Normal images
    X = np.array(X_rows)
    y = np.array(y_rows).reshape(-1,1)
    data = np.concatenate([X, y], axis=1)
    # Augmented images
    X_aug = np.array(X_aug_rows)
    data_augmented = np.concatenate([X_aug, y], axis=1)

    print(f'Extracted: {num_imgs}')
    print(f'Blank Lables: {num_blank}')
    return data, data_augmented

def extract_and_save(dataset_dir, base_dir, out_name):
    # Get all relevant directories
    train_images_dir = os.path.join(dataset_dir, 'train', 'images')
    train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    
    valid_images_dir = os.path.join(dataset_dir, 'valid', 'images')
    valid_labels_dir = os.path.join(dataset_dir, 'valid', 'labels')
    
    test_images_dir = os.path.join(dataset_dir, 'test', 'images')
    test_labels_dir = os.path.join(dataset_dir, 'test', 'labels')
    
    # Extract & augmented images
    train, train_aug = extract(train_labels_dir, train_images_dir,) 
    valid, valid_aug = extract(valid_labels_dir, valid_images_dir,)
    test, _ = extract(test_labels_dir, test_images_dir,)
    
    # Combine train and valid sets together
    train = np.concatenate([train, valid], axis=0)
    train_aug = np.concatenate([train_aug, valid_aug], axis=0)

    np.savez_compressed(os.path.join(base_dir, f'{out_name}_train_extracted'), extracted=train, extracted_augmented=train_aug)
    np.savez_compressed(os.path.join(base_dir, f'{out_name}_test_extracted'), extracted=test)

# Dataset directory was orginally on Kaggle
v8_dir = '/kaggle/input/bone-fracture-detection-computer-vision-project/BoneFractureYolo8'
v4_dir = '/kaggle/input/bone-fracture-detection-computer-vision-project/bone fracture detection.v4-v4.yolov8'

base_dir = os.path.dirname(__file__) # Originally /kaggle/working
extract_and_save(v8_dir, base_dir, 'v8')
extract_and_save(v4_dir, base_dir, 'v4')