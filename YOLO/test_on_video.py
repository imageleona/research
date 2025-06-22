import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('/mnt/c/Users/_s2111724/yolo/runs/detect/train104/weights/best.pt')  

# Path to the video you want to test boug
video_path = "/mnt/c/Users/_s2111724/yolo/wantsu12.mp4"
cap = cv2.VideoCapture(video_path)

annotated_frames = []
while cap.isOpened():  # Loop while frames are available
    ret, frame = cap.read()
    if ret:
        # Lower the confidence threshold to detect objects with low confidence scores
        results = model.predict(source=frame, conf=0.5)  # Adjust 'conf' as needed (e.g., 0.1)
        annotated_frame = results[0].plot()
    else:
        break
    annotated_frames.append(annotated_frame)

# Save the annotated frames as an MP4 video
height, width, layers = annotated_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./wantsu12_bougu.mp4', fourcc, 30, (width, height))
for frame in annotated_frames:
    video.write(frame)
video.release()
cap.release()
cv2.destroyAllWindows()
print('Finished')