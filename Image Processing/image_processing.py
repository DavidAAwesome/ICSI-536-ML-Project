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
        return None

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

def augment_and_save(image, output_dir, base_name, num_augmented=5):
    """
    Apply augmentations to an image and save the generated images using scikit-image.
    """
    for i in range(num_augmented):
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
        augmented_uint8 = (augmented * 255).astype(np.uint8)
        if len(augmented_uint8.shape) == 3 and augmented_uint8.shape[-1] == 1:
            augmented_uint8 = augmented_uint8[:, :, 0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_aug_{i}.png"), augmented_uint8)

if __name__ == "__main__":
    # input and output directories
    input_folder = "input_images"
    output_folder = "augmented_images"

    # Create output folder if dne
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        processed_image = process_image(image_path, convert_to_gray=True)

        if processed_image is not None:
            # Use the file name as a base name for saving
            base_name, _ = os.path.splitext(filename)
            # Convert image values back to [0, 255] for saving with OpenCV
            processed_image_uint8 = (processed_image * 255).astype(np.uint8)
            # If grayscale, remove the extra dimension for saving
            if processed_image_uint8.shape[-1] == 1:
                processed_image_uint8 = processed_image_uint8[:, :, 0]
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_processed.png"), processed_image_uint8)

            # Apply and save augmented images
            augment_and_save(processed_image, output_folder, base_name, num_augmented=5)
