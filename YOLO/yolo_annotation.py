from ultralytics import YOLO

model = YOLO("C:/Users/_s2111724/yolo/runs/detect/train104/weights/best.pt")
results = model.predict(
    source="C:/Users/_s2111724/Documents/yolo_fine_tuning/7",  # a folder of images or a single image
    conf=0.25,                          # confidence threshold
    save=True,                          # save predicted images
    save_txt=True                       # save bounding boxes in YOLO text format
)
