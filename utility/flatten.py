import json
import os

def flatten_json_lstm(input_json_path, output_json_path):
    """
    Reads a JSON file, filters sequences with exactly 30 frames, 
    flattens the data into an LSTM-friendly format, 
    and saves the output as a new JSON file.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the flattened JSON file.

    Returns:
        None
    """
    # Load JSON data from file
    with open(input_json_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    
    sequences = []
    
    for video in json_data.get("data", []):  # Ensure 'data' key exists
        frames = video.get("frames", [])  # Extract frames safely
        if len(frames) == 30:  # Only accept sequences with exactly 30 frames
            sequences.append(frames)

    # Save the flattened data to a new JSON file
    with open(output_json_path, "w", encoding="utf-8") as outfile:
        json.dump({"data": sequences}, outfile, indent=4)

    print(f"Flattened data with 30-frame sequences saved to {output_json_path}")

# Example usage
input_json_path = "C:/Users/_s2111724/dataset_by_moves/keypoints-dataset-by-moves_notflat/dataset_gyakujo.json"   # Replace with your input JSON file path
output_json_path = "C:/Users/_s2111724/dataset_by_moves/keypoints-dataset-by-moves_flat/dataset_gyakujo.json" # Replace with your desired output file path

flatten_json_lstm(input_json_path, output_json_path)