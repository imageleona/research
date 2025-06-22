if __name__ == '__main__':
    from ultralytics import YOLO

    # Load YOLO model
    model = YOLO('yolo11n.pt')

    # Train the model with data augmentation
    model.train(
        data="C:/Users/_s2111724/yolo/bougu_dataset/data.yaml",  # Path to your dataset's YAML
        epochs=300,  # Number of epochs
        batch= 32,    # Batch size
        mixup= 0.3,
        hsv_h=0.015,  # Hue augmentation (fraction)
        hsv_s=0.7,    # Saturation augmentation (fraction)
        hsv_v=0.4,    # Value augmentation (fraction)
        fliplr=0.5,   # Horizontal flip (probability)
        flipud=0.0,   # Vertical flip (probability)
        mosaic=1.0    # Mosaic augmentation (probability)
    )
