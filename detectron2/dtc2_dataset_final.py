import os
import pandas as pd
import cv2
import json
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer

setup_logger()

def visualize_frame(frame, outputs):
    """Optional visualization of a frame with its predictions."""
    v = Visualizer(frame[:, :, ::-1], scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()

def initialize_predictor():
    """Initialize the Detectron2 predictor with the COCO keypoints model."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    return DefaultPredictor(cfg)

def extract_keypoints(outputs):
    """
    Extract keypoints from Detectron2 predictions, considering only
    the person with the highest confidence score.
    Returns a list of x,y pairs (flattened) for that person.
    """
    if len(outputs["instances"].pred_keypoints) == 0:
        return []

    keypoints = outputs["instances"].pred_keypoints
    scores = outputs["instances"].scores  # Confidence scores for each detection

    if len(scores) == 0:
        return []

    # Get index of the person with the highest confidence score
    best_index = scores.argmax().item()

    best_person_keypoints = []
    missing_keypoints = 0

    for keypoint in keypoints[best_index]:
        x, y, score = keypoint.tolist()
        if score > 0.0:  # only keep keypoints with non-zero confidence
            best_person_keypoints.extend([x, y])
        else:
            missing_keypoints += 1

    print(f"Selected Person: Missing keypoints: {missing_keypoints}")
    return [best_person_keypoints]  # List of one person's flattened keypoints

def process_video(video_path, predictor):
    """
    Process a single video file at `video_path` using the predictor.
    Returns a list of keypoints (flattened) for each frame, i.e.
    [
      [frame0_keypoints],
      [frame1_keypoints],
      ...
    ]
    """
    keypoints_sequence = []
    video = cv2.VideoCapture(video_path)
    frame_number = 0

    while True:
        has_frame, frame = video.read()
        if not has_frame:
            break

        outputs = predictor(frame)
        # visualize_frame(frame, outputs)  # Uncomment to visualize
        keypoints = extract_keypoints(outputs)

        print(f"Frame {frame_number}:")
        if keypoints:
            print(f"  Detected persons: {len(keypoints)}")
            for idx, person_kpts in enumerate(keypoints, start=1):
                print(f"    Person {idx}: {len(person_kpts)} values (expected ~34 coords)")
        else:
            print("  No keypoints detected.")

        # Flatten all persons into one list for the frame
        frame_keypoints = []
        for person_kpts in keypoints:
            frame_keypoints.extend(person_kpts)
        keypoints_sequence.append(frame_keypoints)

        print(f"Processed frame {frame_number} of {os.path.basename(video_path)}")
        frame_number += 1

    video.release()
    return keypoints_sequence

def main():
    # Input Excel file and output JSON path
    excel_path = "/home/appuser/detectron2-code/video-data-by-moves/file_list_wantsu.xlsx"
    output_json_path = "/home/appuser/detectron2-code/keypoints-dataset-by-moves/dataset_wantsu.json"

    # Read the Excel file
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Ensure we have a 'Filename' column
    if "Filename" not in df.columns:
        print("Error: Excel file must contain a 'Filename' column.")
        return

    # Initialize Detectron2 predictor
    predictor = initialize_predictor()

    # Prepare final output dict, matching YOLO-like structure
    # 'index' is just a fixed string or can be changed as needed
    output_data = {
        "index": "01",
        "data": []
    }

    # Process each video in the Excel file
    for _, row in df.iterrows():
        video_path = str(row["Filename"])

        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue

        if video_path == excel_path:
            continue  # Skip if the Excel references itself

        # Extract keypoints for all frames
        keypoints_sequence = process_video(video_path, predictor)

        # Append to output_data in the same shape as YOLO code
        # i.e. each video object has video_path + frames list
        video_entry = {
            "video_path": video_path,
            "frames": keypoints_sequence
        }
        output_data["data"].append(video_entry)

    # Save to JSON
    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Keypoints data saved to {output_json_path}")

if __name__ == "__main__":
    main()
