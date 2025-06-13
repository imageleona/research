import json
import os

def process_json_file(input_filepath, output_directory):
    """
    Loads a JSON file, truncates each sequence in the 'data' field to 15 frames,
    and saves the modified data to a new JSON file.
    """
    try:
        with open(input_filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_filepath}: {e}")
        return
    except FileNotFoundError:
        print(f"File not found: {input_filepath}")
        return

    # Ensure 'data' key exists and is a list
    if "data" in data and isinstance(data["data"], list):
        processed_sequences = []
        for i, sequence in enumerate(data["data"]):
            if isinstance(sequence, list) and len(sequence) >= 15:
                # Take the first 15 frames of the current sequence
                processed_sequences.append(sequence[:15])
            elif isinstance(sequence, list) and len(sequence) < 15:
                print(f"Warning: Sequence {i+1} in {input_filepath} has fewer than 15 frames ({len(sequence)}). Keeping all available frames for this sequence.")
                processed_sequences.append(sequence)
            else:
                print(f"Warning: Unexpected format for sequence {i+1} in {input_filepath}. Skipping or including as-is.")
                processed_sequences.append(sequence) # Include it as-is if format is unexpected

        data["data"] = processed_sequences
    else:
        print(f"Warning: 'data' key not found or not a list in {input_filepath}. Skipping processing for this file.")
        return

    # Construct output file path
    filename = os.path.basename(input_filepath)
    output_filepath = os.path.join(output_directory, f"15_frames_{filename}")

    # Save the modified data to a new JSON file
    try:
        with open(output_filepath, 'w') as f:
            json.dump(data, f, indent=4) # 'indent=4' makes the output JSON readable
        print(f"Successfully processed and saved: {output_filepath}")
    except IOError as e:
        print(f"Error writing to {output_filepath}: {e}")

# --- Configuration ---
# Set the path to your original JSON files
input_json_directory = "C:/Users/_s2111724/utility/keypoints_dataset_by_moves"

# Set the path where you want to save the new 15-frame JSON files
output_json_directory = "C:/Users/_s2111724/utility/keypoints_dataset_by_moves_15frames"
# ---------------------

# Create the output directory if it doesn't exist
os.makedirs(output_json_directory, exist_ok=True)

# Iterate through all JSON files in the input directory
for filename in os.listdir(input_json_directory):
    if filename.endswith(".json"):
        input_filepath = os.path.join(input_json_directory, filename)
        process_json_file(input_filepath, output_json_directory)

print("\nDataset processing complete!")