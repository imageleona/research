import json
import numpy as np
from collections import Counter

def check_data_shapes(json_file):
    """
    Loads the combined JSON file and checks the shape of each sample in data["data"].
    Prints whether all samples share the same shape or not, and lists shape frequencies.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    shapes = []
    for i, sample in enumerate(data["data"]):
        # Convert the sample into a numpy array for shape checking
        arr = np.array(sample)
        shapes.append(arr.shape)

    # Count how many samples have each distinct shape
    shape_counts = Counter(shapes)

    if len(shape_counts) == 1:
        # Only one unique shape
        unique_shape = next(iter(shape_counts))
        print(f"All {len(shapes)} samples have the SAME shape: {unique_shape}")
    else:
        # Multiple shapes found
        print(f"Found {len(shape_counts)} different shapes among {len(shapes)} samples:")
        for shape, count in shape_counts.items():
            print(f"  Shape {shape} appears {count} times.")

if __name__ == "__main__":
    # Example usage:
    json_file = "C:/Users/_s2111724/training/data5_skeleton/skeleton_yuko_flat.json"
    check_data_shapes(json_file)