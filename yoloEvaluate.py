"""We evaluate the trained model on hte validation split of the VisDrone Dataset.
Because training my own model was not possible in google colab, and took very long time on my local sytem, 
I have downloaded an already trained yolov8m on the VisDrone drom the internet: https://huggingface.co/mshamrai/yolov8m-visdrone.
I will evaluate this model.
"""

from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark


def main():
    model = YOLO("best.pt")

    # Validate the model
    metrics = model.val(data="data.yaml", imgsz=640, batch=4, device=0, save_json=True)
    # metrics.box.map  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps  # a list containing mAP50-95 for each category
    # metrics.box.image_metrics  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN

    # Benchmark on GPU
    # benchmark(model="best.pt", data="data.yaml", imgsz=640, half=False, int8=False, device=0, verbose=True, format="-")
    # benchmark(model="best.pt", data="data.yaml", imgsz=640, half=False, int8=False, device=0, verbose=True, format="torchscript")
    # benchmark(model="best.pt", data="data.yaml", imgsz=640, half=False, int8=False, device=0, verbose=True, format="onnx")
    # benchmark(model="best.pt", data="data.yaml", imgsz=640, half=False, int8=False, device=0, verbose=True, format="engine")

if __name__ == "__main__":
    main()

