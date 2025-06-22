import cv2
import os
import torch
from detectron2.config import get_cfg
# Changed the import style for model_zoo for potentially better compatibility
import detectron2.model_zoo as model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import time

# --- Configuration ---
# Path to your input video file inside the container
# Make sure this path is correct based on your volume mapping
input_video_path = "/home/appuser/detectron2-code/uchikomi0.mp4" # <--- CHANGE THIS
# Path for the output video file
output_video_path = "/home/appuser/detectron2-code/uchikomi0_with_skeleton.mp4" # <--- CHANGE THIS

# Choose a Detectron2 model configuration file for keypoint detection
# You can find available models in the Detectron2 Model Zoo:
# https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
# Example using a R50-FPN Mask R-CNN with Keypoints:
config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"

# Threshold for detection scores (adjust as needed)
confidence_threshold = 0.7

# --- Setup Detectron2 Predictor ---
cfg = get_cfg()
# Add project-specific config (e.g., from original Detectron2 repo)
# This line might need adjustment based on where the detectron2 configs are located in your container
# Using the imported model_zoo directly
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold  # set threshold for this model
# Find a model from the detectron2 model zoo. You can use the URL or a local path
# Using the imported model_zoo directly
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

predictor = DefaultPredictor(cfg)

# --- Video Processing ---
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4

# Setup video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Get metadata for visualization
# This assumes 'coco_2017_train' dataset is registered and has skeleton info
# If you are using a custom dataset, you might need to register it first
# or manually define the skeleton structure as shown in the previous example code you shared.
try:
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
except AttributeError:
     print("Warning: Dataset metadata not found or does not have TRAIN attribute. Using default COCO metadata.")
     # Fallback to standard COCO metadata if cfg.DATASETS.TRAIN[0] is not available
     metadata = MetadataCatalog.get("coco_2017_train")
     # You might need to manually define metadata.skeleton here if it's not in the registered data
     if not hasattr(metadata, 'skeleton') or metadata.skeleton is None:
          print("Warning: Skeleton not found in metadata. Manually defining COCO skeleton.")
          metadata.skeleton = [
             (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
             (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
             (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
             (1, 3), (2, 4)
         ]


print("Starting video processing...")
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    # Ensure frame is in the correct format (BGR for OpenCV, but Detectron2 expects RGB internally)
    # The Visualizer handles the BGR to RGB conversion when initialized with BGR input
    outputs = predictor(frame)

    # Draw visualizations
    # Initialize Visualizer with RGB image (frame[:, :, ::-1])
    v = Visualizer(frame[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    # Use InstanceMode.SEGMENTATION for masks if available, or remove for just keypoints/boxes
    # The output from draw_instance_predictions is RGB, convert back to BGR for OpenCV
    out_frame = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

    # Write the frame to the output video
    out.write(out_frame[:, :, ::-1]) # Convert back to BGR for OpenCV

    frame_count += 1
    # Print progress every second of video processed
    if fps > 0 and frame_count % fps == 0:
        print(f"Processed {frame_count} frames...")
    # Handle case where fps is 0 or unknown
    elif fps <= 0 and frame_count % 100 == 0:
         print(f"Processed {frame_count} frames...")


end_time = time.time()
print(f"Finished processing {frame_count} frames.")
print(f"Total processing time: {end_time - start_time:.2f} seconds")

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_video_path}")
