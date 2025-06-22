import os
import pandas as pd
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

setup_logger()

# Set the output directory for videos
OUTPUT_DIR = "/home/appuser/detectron2-code/output-videos4"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create directory if it doesn't exist

def initialize_predictor():
    """Initialize the Detectron2 predictor with the COCO keypoints model."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # detection threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    return DefaultPredictor(cfg)

def process_video(video_path, predictor):
    """
    Processes a video frame by frame, applies pose estimation, and saves an output video.
    This version manually draws lines connecting keypoints based on a defined skeleton.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the output video path in OUTPUT_DIR
    video_filename = os.path.basename(video_path)
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(video_filename)[0] + "_pose_estimation.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Retrieve metadata and define the skeleton for keypoint connections.
    metadata = MetadataCatalog.get("coco_2017_train")
    # (The keypoint indices here are 0-indexed.)
    metadata.skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4)
    ]

    frame_number = 0
    # Confidence threshold for drawing a line between two keypoints.
    kp_threshold = 0.05  # adjust if needed

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break  # End of video

        # Run pose estimation
        outputs = predictor(frame)

        # Draw the standard predictions (boxes, keypoints, etc.) with the Visualizer.
        v = Visualizer(frame[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # Convert the image back to BGR format and ensure it is C-contiguous.
        output_frame = np.ascontiguousarray(v.get_image()[:, :, ::-1])

        # Now add skeleton lines manually.
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_keypoints"):
            # Shape: (N, num_keypoints, 3) where last dim is [x, y, score]
            keypoints = instances.pred_keypoints.numpy()
            for person in keypoints:
                # Loop over each connection defined in metadata.skeleton
                for (start_idx, end_idx) in metadata.skeleton:
                    pt1 = person[int(start_idx)]
                    pt2 = person[int(end_idx)]
                    # Only draw if both keypoints have a confidence greater than the threshold.
                    if pt1[2] > kp_threshold and pt2[2] > kp_threshold:
                        cv2.line(
                            output_frame,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            color=(0, 255, 0),  # green line; change as desired
                            thickness=2         # line thickness; change as desired
                        )

        # Write the annotated frame to the output video.
        out.write(output_frame)
        print(f"Processed frame {frame_number} for {video_filename}")
        frame_number += 1

    cap.release()
    out.release()
    print(f"Processed video saved at: {output_path}")

def main():
    # Input Excel file path
    excel_path = "/home/appuser/detectron2-code/video-data-by-moves/kizami/file_list_kizami.xlsx"

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

    # Process each video in the Excel file
    for _, row in df.iterrows():
        video_path = str(row["Filename"])
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        process_video(video_path, predictor)

if __name__ == "__main__":
    main()