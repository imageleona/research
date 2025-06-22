import cv2
import json
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('C:/Users/_s2111724/yolo/runs/detect/train86/weights/best.pt')  # Path to best.pt

# Path to the video
video_path = "C:/Users/_s2111724/Documents/karate-clip3/IMG_7630_kizami_muko_clip_015.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize variables
annotated_frames = []
annotations = {"index": "01", "data": []}  # Match the format of the provided JSON
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.track(frame, persist=True, conf=0.01)
    annotated_frame = results[0].plot()
    annotated_frames.append(annotated_frame)

    # Process each detected object
    frame_data = []
    for result in results[0].boxes:
        box_points = result.xyxy[0].cpu().numpy()
        frame_data.append(box_points.tolist())  # Add points to the frame array

    # Ensure two objects per frame
    while len(frame_data) < 2:
        frame_data.append([0] * 4)  # Append zeros for missing objects

    # Concatenate the two objects' bounding box coordinates into one array
    combined_data = frame_data[0] + frame_data[1]  # Combine the first two objects' coordinates

    # Add the combined data to the annotations
    annotations["data"].append(combined_data)  # Append as a single array

    frame_count += 1

# Save the annotations as a JSON file
output_json_path = './annotations.json'
with open(output_json_path, 'w') as f:
    json.dump(annotations, f, indent=4)

cap.release()
cv2.destroyAllWindows()

print(f'Annotations saved as {output_json_path}')