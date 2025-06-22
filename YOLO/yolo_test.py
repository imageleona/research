import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("C:/Users/_s2111724/yolo/runs/detect/train104/weights/best.pt")

# Open the video file
video_path = "C:/Users/_s2111724/Documents/karate-clip3/IMG_7635_kizami_yuko_clip_015.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.1)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()