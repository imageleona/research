import json
import random

def split_dataset(input_json, output_train, output_test, seed=None):
    """
    Splits a dataset into 5/6 train and 1/6 test.
    
    Args:
        input_json (str): Path to the input JSON file.
        output_train (str): Path to save the training JSON file.
        output_test (str): Path to save the test JSON file.
        seed (int, optional): Random seed for reproducibility. Default is None.
    """
    if seed is not None:
        random.seed(seed)
    
    # Load data
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Shuffle the samples (in-place)
    random.shuffle(data["data"])

    # Compute split indexes
    total_samples = len(data["data"])
    test_size = total_samples // 6   # integer division for 1/6 test
    train_size = total_samples - test_size

    # Split the data
    train_data = data["data"][:train_size]
    test_data = data["data"][train_size:]
    
    # Create new dictionaries for train/test with the same "index" field
    train_dict = {"index": data["index"], "data": train_data}
    test_dict = {"index": data["index"], "data": test_data}

    # Save to output JSON files
    with open(output_train, 'w') as f:
        json.dump(train_dict, f, indent=4)
    with open(output_test, 'w') as f:
        json.dump(test_dict, f, indent=4)

    print(f"Split completed: {train_size} samples for training, {test_size} samples for testing.")

if __name__ == "__main__":
    # Example usage
    # Provide the paths to your dataset, train output, and test output files
    input_file = "C:/Users/_s2111724/dataset_by_moves/keypoints-dataset-by-moves_flat/dataset_gyakujo.json"
    output_train_file = "C:/Users/_s2111724/dataset_by_moves/keypoints-dataset-by-moves_flat_train/dataset_gyakujo_train.json"
    output_test_file = "C:/Users/_s2111724/dataset_by_moves/keypoints-dataset-by-moves_flat_test/dataset_gyakujo_test.json"

    # Optional: set a random seed for reproducibility
    random_seed = 42

    # Run the split
    split_dataset(input_file, output_train_file, output_test_file, seed=random_seed)