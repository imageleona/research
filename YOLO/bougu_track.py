import cv2
from ultralytics import YOLO

# 1) Load your fine-tuned YOLO model
model = YOLO('C:/Users/_s2111724/yolo/runs/detect/train104/weights/best.pt')

# 2) Path to the video
video_path = "C:/Users/_s2111724/Documents/karate-clip3/IMG_7635_kizami_yuko_clip_015.mp4"
cap = cv2.VideoCapture(video_path)

annotated_frames = []

while cap.isOpened():  
    ret, frame = cap.read()
    if not ret:
        break

    # 3) Perform detection + MOT tracking with a chosen tracker config
    #    e.g. "bytetrack.yaml" or "strongsort.yaml"
    results = model.predict(
        source=frame,
        conf=0.1,         # Adjust threshold for low-confidence detections     
        #tracker="bytetrack.yaml"
    )

    # 4) Plot detections/tracks on the current frame
    #    This automatically draws bounding boxes, track IDs, etc.
    annotated_frame = results[0].plot()
    annotated_frames.append(annotated_frame)

# 5) Save the annotated frames as an MP4 video
height, width, layers = annotated_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = './annotated_video16.mp4'
video = cv2.VideoWriter(out_path, fourcc, 30, (width, height))

for frame in annotated_frames:
    video.write(frame)

video.release()
cap.release()
cv2.destroyAllWindows()
print(f'Finished writing {out_path}')