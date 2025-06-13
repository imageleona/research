import json
import os

def load_json(json_path):
    """Load a JSON file and return its content."""
    try:
        with open(json_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None

def save_json(data, output_path):
    """Save the combined data to a JSON file."""
    try:
        with open(output_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Combined JSON saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON file {output_path}: {e}")

def extract_filename_no_ext(path):
    """
    Extract the base filename (without directory path and extension).
    E.g. 'C:/videos/clip1.mp4' -> 'clip1'
    """
    base = os.path.basename(path)         # e.g. "clip1.mp4"
    name, _ = os.path.splitext(base)      # ("clip1", ".mp4")
    return name

def fuse_json_files_by_video(object_json_path, skeleton_json_path, output_combined_json_path):
    """
    Fuse object data and skeleton data from two JSON files into the desired format:
    {
        "data": [
            [[frame1], [frame2], ..., [frame30]],  # combined frames for video1
            [[frame1], [frame2], ..., [frame30]],  # combined frames for video2
            ...
        ]
    }
    Only keeps videos that have exactly 30 frames in both object and skeleton data.
    """

    # 1. Load the JSON files
    object_data = load_json(object_json_path)
    skeleton_data = load_json(skeleton_json_path)

    if not object_data or not skeleton_data:
        print("Error: Could not load one or both JSON files.")
        return

    # 2. Create a mapping of filename -> skeleton frames
    #    Use the filename WITHOUT extension as the key.
    skeleton_map = {
        extract_filename_no_ext(entry["video_path"]): entry["frames"]
        for entry in skeleton_data["data"]
    }

    combined_data = {
        "data": []
    }

    # 3. Iterate over each object entry
    for obj_entry in object_data["data"]:
        obj_filename = extract_filename_no_ext(obj_entry["video_path"])
        obj_frames = obj_entry["frames"]

        # Debug: print info on the object side
        print(f"\n[DEBUG] Object video: {obj_filename}, frames: {len(obj_frames)}")

        # Check if skeleton side has this video
        if obj_filename not in skeleton_map:
            print(f"Warning: Skeleton data not found for {obj_filename}")
            continue

        ske_frames = skeleton_map[obj_filename]

        # Debug: print info on the skeleton side
        print(f"[DEBUG] Skeleton video: {obj_filename}, frames: {len(ske_frames)}")

        # 4. Check if the number of frames is exactly 30 on both sides
        if len(obj_frames) == 30 and len(ske_frames) == 30:
            # Combine frames
            combined_frames = [of + sf for of, sf in zip(obj_frames, ske_frames)]
            print(f"[DEBUG] Combined frames for {obj_filename}: {len(combined_frames)}")

            combined_data["data"].append(combined_frames)
        else:
            # We skip this video if it doesn't have exactly 30 frames in both
            print(f"Skipping {obj_filename} because it does not have exactly 30 frames.")

    # 5. Save the combined JSON
    save_json(combined_data, output_combined_json_path)

def main():
    # Example paths (update these to your actual file locations):
    object_json_path = "C:/Users/_s2111724/yolo/yolo_output/karate_bougu_yuko.json"
    skeleton_json_path = "C:/Users/_s2111724/detectron2-code/keypoints-dataset/dataset_yuko.json"
    output_combined_json_path = "C:/Users/_s2111724/combined/karate_combined_yuko2.json"

    fuse_json_files_by_video(object_json_path, skeleton_json_path, output_combined_json_path)

if __name__ == "__main__":
    main()