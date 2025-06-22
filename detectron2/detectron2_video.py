import os
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

def setup_predictor():
    """
    Initialize the Detectron2 predictor using the COCO-Keypoints model.
    """
    cfg = get_cfg()
    # Use the keypoint RCNN R101-FPN 3x configuration.
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    # Set a detection threshold (adjust as needed)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Download weights from model zoo.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor

def annotate_frame(frame, predictor, metadata, kp_threshold=0.05):
    """
    Run pose estimation on a single frame, draw predictions using Detectron2's Visualizer,
    and manually add skeleton lines between keypoints.
    
    Args:
        frame (np.ndarray): The input image frame (BGR).
        predictor (DefaultPredictor): The Detectron2 predictor.
        metadata: MetadataCatalog entry (should include 'skeleton').
        kp_threshold (float): Minimum confidence for a keypoint to be used in drawing a line.
    
    Returns:
        np.ndarray: The annotated frame (BGR).
    """
    # Run the model to get predictions
    outputs = predictor(frame)

    # Use Visualizer to draw the standard instance predictions (boxes, keypoints, etc.)
    v = Visualizer(frame[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # Convert the visualized image back to BGR and ensure the array is contiguous.
    annotated_frame = np.ascontiguousarray(v.get_image()[:, :, ::-1])
    
    # Check if there are keypoints in the predictions.
    if outputs["instances"].has("pred_keypoints"):
        # The shape of keypoints is (N, num_keypoints, 3) where each keypoint is (x, y, score)
        keypoints = outputs["instances"].pred_keypoints.cpu().numpy()
        for person in keypoints:
            # Loop over each connection defined in the metadata skeleton
            for (start_idx, end_idx) in metadata.skeleton:
                pt1 = person[int(start_idx)]
                pt2 = person[int(end_idx)]
                # Only draw the line if both keypoints have confidence greater than the threshold
                if pt1[2] > kp_threshold and pt2[2] > kp_threshold:
                    cv2.line(
                        annotated_frame,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        color=(0, 255, 0),  # Green color for skeleton lines
                        thickness=2
                    )
    return annotated_frame

def process_video(input_video_path, output_video_path, predictor, metadata):
    """
    Process the input video frame by frame, annotate each frame with pose estimation,
    and save the output video.
    
    Args:
        input_video_path (str): Path to the input video.
        output_video_path (str): Path where the output video will be saved.
        predictor (DefaultPredictor): The Detectron2 predictor.
        metadata: MetadataCatalog entry containing the skeleton.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {input_video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create the video writer using MP4 codec.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame = annotate_frame(frame, predictor, metadata)
        out.write(annotated_frame)
        
        print(f"Processed frame {frame_number}")
        frame_number += 1

    cap.release()
    out.release()
    print(f"Output video saved to: {output_video_path}")

def main():
    # Update these paths as necessary.
    input_video_path = "/home/appuser/detectron2-code/wantsu12_bougu.mp4"      # Path to your input video.
    output_video_path = "/home/appuser/detectron2-code/wantsu12_bougu_skeleton.mp4" # Path for the output video.
    
    # Initialize the predictor.
    predictor = setup_predictor()
    
    # Retrieve the metadata for COCO training set and define the skeleton.
    metadata = MetadataCatalog.get("coco_2017_train")
    # Define the skeleton as pairs of keypoint indices (0-indexed).
    metadata.skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4)
    ]
    
    process_video(input_video_path, output_video_path, predictor, metadata)

if __name__ == "__main__":
    main()