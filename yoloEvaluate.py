"""We evaluate the trained model on hte validation split of the VisDrone Dataset.
Because training my own model was not possible in google colab, and took very long time on my local sytem, 
I have downloaded an already trained yolov8m on the VisDrone drom the internet: https://huggingface.co/mshamrai/yolov8m-visdrone.
I will evaluate this model.
"""

from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import argparse
from prettytable import PrettyTable
import numpy as np


def getParser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modelPath', required=True, help="The .pt model file to evaluate.")
    parser.add_argument("--dataFilePath", type=str, default="data.yaml", help="The .yaml file that describes the dataset.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "test"], help="The split to evaluate on.")
    parser.add_argument("--imageSize", type=int, default=640, help="The image size that the model uses internally.")
    parser.add_argument("--batchSize", type=int, default=4, help="The batch size for evaluation.")
    
    return parser


def main():

    parser = getParser()
    args = parser.parse_args()
    
    model = YOLO(args.modelPath)

    # Validate the model
    metrics = model.val(data=args.dataFilePath, imgsz=args.imageSize, batch=args.batchSize, device=0, save_json=False, split=args.split)
    
    # Print metrics
    allAPTable = PrettyTable(["Class", "AP50", "AP55", "AP60", "AP65", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95"])
    for i, cls in metrics.names.items():
        allAPTable.add_row([cls] + np.round(metrics.box.all_ap[i], 5).tolist())
    allAPTable.title = "All AP"
    print(allAPTable)
    
    ap5095Table = PrettyTable(["Class", "AP50-95"])
    for i, cls in metrics.names.items():
        ap5095Table.add_row([cls, np.round(metrics.box.ap[i], 5)])
    ap5095Table.title = "AP50-95 per Class"
    print(ap5095Table)
    
    print(f"mAP50: {metrics.box.map50}\n")
    print(f"mAP75: {metrics.box.map75}\n")
    print(f"mAP50-95: {metrics.box.map}\n")


if __name__ == "__main__":
    main()

