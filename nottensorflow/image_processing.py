import os
import cv2
import numpy as np
from skimage import transform
from skimage.util import random_noise

# Desired image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

def process_image(image_path, convert_to_gray=True):
    """
    - Optionally converts to grayscale
    - Resizes to fixed dimensions (224x224)
    - Normalizes pixel values to [0,1]
    """
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        raise RuntimeError()

    # Convert image to grayscale if needed
    if convert_to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Expand dims to keep channel dimension (required for processing: height x width x 1)
        img = np.expand_dims(img, axis=-1)
    else:
        # Convert BGR to RGB if keeping color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    # Normalize pixel values to the range [0,1]
    img = img.astype("float32") / 255.0
    return img

def augment_and_save(image, output_dir, base_name, save=True):
    """
    Apply randomized augmentations to an image and save the generated images using scikit-image.
    - `image`: numpy array
    """
    # Apply random augmentations
    augmented = image.copy()
    
    # Random rotation (-15 to 15 degrees)
    angle = np.random.uniform(-15, 15)
    augmented = transform.rotate(augmented, angle, mode='reflect')
    
    # Random zoom
    zoom = np.random.uniform(0.9, 1.1)
    channel_axis = -1 if len(image.shape) > 2 else None
    augmented = transform.rescale(augmented, zoom, channel_axis=channel_axis)
    augmented = transform.resize(augmented, image.shape[:2])
    
    # Random horizontal flip
    if np.random.random() > 0.5:
        augmented = np.fliplr(augmented)
    
    # Random brightness adjustment
    brightness_factor = np.random.uniform(0.8, 1.2)
    augmented = augmented * brightness_factor
    augmented = np.clip(augmented, 0, 1)
    
    # Add slight random noise
    augmented = random_noise(augmented, mode='gaussian', var=0.01)
    
    # Save the augmented image
    img_path = os.path.join(output_dir, f"{base_name}_aug.png") if save else ""
    if save:
        augmented_uint8 = (augmented * 255).astype(np.uint8)
        if len(augmented_uint8.shape) == 3 and augmented_uint8.shape[-1] == 1:
            augmented_uint8 = augmented_uint8[:, :, 0]
        cv2.imwrite(img_path, augmented_uint8)
    return augmented, img_path
