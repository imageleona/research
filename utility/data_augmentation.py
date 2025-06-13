import json
import random
import numpy as np

def augment_sample(sample, image_width, image_height):
    """
    Augments a single sample by applying consistent transformations to all frames.

    Args:
        sample (list): List of frames, where each frame contains flattened keypoints.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        list: Augmented sample with the same number of frames.
    """
    # Decide augmentations for the entire sample
    flip = random.random() > 0.5
    scale = random.uniform(0.8, 1.2)  # Scale by 0.8x to 1.2x
    angle = random.uniform(-15, 15)  # Rotate by -15 to 15 degrees
    tx = random.uniform(-0.1 * image_width, 0.1 * image_width)  # Translate by -10% to 10% of width
    ty = random.uniform(-0.1 * image_height, 0.1 * image_height)  # Translate by -10% to 10% of height

    augmented_sample = []
    for frame in sample:
        xy_coords = [(frame[i], frame[i + 1]) for i in range(0, len(frame), 2)]
        
        # Apply horizontal flip
        if flip:
            xy_coords = [(image_width - x, y) for x, y in xy_coords]

        # Apply scaling
        xy_coords = [(x * scale, y * scale) for x, y in xy_coords]

        # Apply rotation
        radians = np.radians(angle)
        cos_theta, sin_theta = np.cos(radians), np.sin(radians)
        cx, cy = image_width / 2, image_height / 2  # Image center
        xy_coords = [
            (
                cos_theta * (x - cx) - sin_theta * (y - cy) + cx,
                sin_theta * (x - cx) + cos_theta * (y - cy) + cy
            )
            for x, y in xy_coords
        ]

        # Apply translation
        xy_coords = [(x + tx, y + ty) for x, y in xy_coords]

        # Flatten back to [x1, y1, x2, y2, ...]
        augmented_sample.append([coord for xy in xy_coords for coord in xy])
    return augmented_sample

def multiply_dataset(json_file, output_file, image_width, image_height, multiplier=90):
    """
    Multiplies the dataset to approximately match the desired size.

    Args:
        json_file (str): Path to the input JSON file.
        output_file (str): Path to save the augmented dataset.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        multiplier (int): Factor by which to multiply the dataset.
    """
    # Load original dataset
    with open(json_file, 'r') as f:
        data = json.load(f)

    original_samples = len(data["data"])
    target_samples = original_samples * multiplier

    print(f"Original samples: {original_samples}, Target samples: {target_samples}")

    augmented_data = {"index": data["index"], "data": []}
    
    # Add original samples
    augmented_data["data"].extend(data["data"])

    # Generate augmented samples
    while len(augmented_data["data"]) < target_samples:
        for sample in data["data"]:
            if len(augmented_data["data"]) >= target_samples:
                break
            augmented_sample = augment_sample(sample, image_width, image_height)
            augmented_data["data"].append(augmented_sample)
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(augmented_data, f, indent=4)
    print(f"Augmented dataset saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    # Input JSON file and output augmented file
    json_file = "C:/Users/_s2111724/dataset_by_moves/keypoints-dataset-by-moves_flat_train_before_augmentation/dataset_gyakujo_train.json"
    output_file = "C:/Users/_s2111724/dataset_by_moves/keypoints_dataset_final/k0004_gyakujo/k0004_gyakujo_train.json"
    
    # Define image dimensions (replace with actual values)
    image_width, image_height = 1920, 1080
    
    # Run augmentation
    multiply_dataset(json_file, output_file, image_width, image_height, multiplier=90)
