import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('/mnt/c/Users/_s2111724/yolo/runs/detect/train104/weights/best.pt')  

# Path to the video
video_path = "/mnt/c/Users/_s2111724/yolo/kizami_yuko_for_path.mp4"
cap = cv2.VideoCapture(video_path)

# Dictionary to store object paths (using class ID as key)
object_paths = {}
MAX_HISTORY = 50  # Maximum number of points to remember per object

annotated_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model.predict(source=frame, conf=0.5)
    detections = results[0].boxes.data
    
    # Temporary dictionary for current frame's objects
    current_objects = {}
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        class_id = int(cls)
        current_objects[class_id] = (center_x, center_y)
        
        # Draw current position
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Update paths for persistent objects
    for class_id in list(object_paths.keys()):
        if class_id in current_objects:
            # Add new point if object is detected
            object_paths[class_id].append(current_objects[class_id])
            # Trim to max history
            if len(object_paths[class_id]) > MAX_HISTORY:
                object_paths[class_id].pop(0)
        else:
            # Remove old paths if object disappears
            del object_paths[class_id]
    
    # Add new objects
    for class_id in current_objects:
        if class_id not in object_paths:
            object_paths[class_id] = [current_objects[class_id]]
    
    # Draw paths for all objects
    for class_id, path in object_paths.items():
        # Draw connecting lines
        for i in range(1, len(path)):
            cv2.line(frame, path[i-1], path[i], (0, 255, 0), 2)
        # Draw small circle at latest position
        if len(path) > 0:
            cv2.circle(frame, path[-1], 4, (0, 255, 255), -1)
    
    annotated_frames.append(frame)

# Save the video
height, width, layers = annotated_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./kizami_yuko_for_path_path.mp4', fourcc, 30, (width, height))
for frame in annotated_frames:
    video.write(frame)
video.release()
cap.release()
cv2.destroyAllWindows()
print('Finished')