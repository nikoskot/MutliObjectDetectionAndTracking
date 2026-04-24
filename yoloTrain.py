from ultralytics import YOLO

def main():
    model = YOLO('yolov8m.pt')  # downloads pretrained COCO weights automatically
            
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=1280,
        batch=8,
        device=0,
        name='detector_v1',
        amp=False,        # enable on Colab
        patience=20,      # early stopping if val metrics plateau
        save_period=10,   # save checkpoint every 10 epochs
        workers=4,        # dataloader workers
    )
    
if __name__ == "__main__":
    main()