import os
import json
import cv2
import pandas as pd
from ultralytics import YOLO

def wsl_path_to_windows(path: str) -> str:
    """
    Convert a WSL path like '/mnt/c/Users/...' to 'C:/Users/...'.
    If the path doesn't start with '/mnt/', returns unchanged.
    """
    if path.startswith("/mnt/"):
        drive_letter = path[5]
        remainder = path[7:]
        new_path = f"{drive_letter.upper()}:/{remainder}"
        return new_path.replace("\\", "/")
    else:
        return path

def initialize_model(model_path):
    """Load a fine-tuned YOLO model from 'model_path'."""
    model = YOLO(model_path)
    return model

def extract_two_centers_from_results(results, target_class_id=0):
    """
    Always return exactly 2 object centers per frame:
      - If 0 found, both centers are [0, 0].
      - If 1 found, second center is [0, 0].
      - If >=2 found, only the top 2 by confidence are used.
    Each center is [cx, cy].
    So the final output is a 4-value list: [cx1, cy1, cx2, cy2].
    """

    # Gather all bounding boxes for the desired class (default: person=0)
    bboxes = []
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if cls_id == target_class_id and conf > 0.0:
                bboxes.append((conf, x1, y1, x2, y2))

    # Sort by descending confidence
    bboxes.sort(key=lambda x: x[0], reverse=True)

    def center_coords(x1, y1, x2, y2):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return [cx, cy]

    if len(bboxes) == 0:
        # No detections => 2 zero placeholders
        return [0, 0, 0, 0]
    elif len(bboxes) == 1:
        # 1 detection => second center is zeros
        conf1, x1, y1, x2, y2 = bboxes[0]
        center1 = center_coords(x1, y1, x2, y2)
        center2 = [0, 0]
        return center1 + center2
    else:
        # 2+ detections => top 2
        conf1, x1a, y1a, x2a, y2a = bboxes[0]
        conf2, x1b, y1b, x2b, y2b = bboxes[1]
        center1 = center_coords(x1a, y1a, x2a, y2a)
        center2 = center_coords(x1b, y1b, x2b, y2b)
        return center1 + center2

def process_video(video_path, model):
    """
    Reads a video frame by frame using OpenCV.
    For each frame, run YOLO -> always output 2 centers (padded/truncated).
    Returns a list with (#frames) elements, each a list of 4 floats:
      [cx1, cy1, cx2, cy2].
    """
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO predictions
        results = model.predict(source=frame, conf=0.1)
        centers = extract_two_centers_from_results(results, target_class_id=0)

        print(f"Frame {frame_number} of {os.path.basename(video_path)}: {centers}")
        frame_results.append(centers)
        frame_number += 1

    cap.release()
    return frame_results

def main():
    # 1) Paths
    excel_path = "C:/Users/_s2111724/utility/list_of_files/file_list_yuko.xlsx"
    output_json_path = "C:/Users/_s2111724/yolo/yolo_output/karate_yuko.json"

    # 2) Read Excel
    df = pd.read_excel(wsl_path_to_windows(excel_path))
    if "Filename" not in df.columns:
        print("Excel file must contain a 'Filename' column.")
        return

    # 3) Load YOLO model
    model_path = "C:/Users/_s2111724/yolo/runs/detect/train108/weights/best.pt"
    model_path = wsl_path_to_windows(model_path)
    model = initialize_model(model_path)

    # 4) Build dataset
    dataset = {
        "index": "01",
        "data": []
    }

    # 5) Process each video
    for _, row in df.iterrows():
        linux_path = str(row["Filename"])
        win_path = wsl_path_to_windows(linux_path)

        if not os.path.exists(win_path):
            print(f"File not found: {linux_path}")
            continue

        frame_results = process_video(win_path, model)
        dataset["data"].append({
            "video_path": linux_path,
            "frames": frame_results
        })

    # 6) Save to JSON
    json_out = wsl_path_to_windows(output_json_path)
    with open(json_out, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Saved 2-object centers to {json_out}")

if __name__ == "__main__":
    main()
