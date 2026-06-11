"""We evaluate the trained model on hte validation split of the VisDrone Dataset.
Because training my own model was not possible in google colab, and took very long time on my local sytem, 
I have downloaded an already trained yolov8m on the VisDrone drom the internet: https://huggingface.co/mshamrai/yolov8m-visdrone.
I will evaluate this model.
"""

from ultralytics import YOLO
import argparse
from prettytable import PrettyTable
import numpy as np
import json
import struct
from pathlib import Path
import shutil
import os

def getParser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modelPath', required=True, help="The .pt model file to evaluate.")
    parser.add_argument("--dataFilePath", type=str, default="data.yaml", help="The .yaml file that describes the dataset.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "test"], help="The split to evaluate on.")
    parser.add_argument("--imageSize", type=int, default=640, help="The image size that the model uses internally.")
    parser.add_argument("--batchSize", type=int, default=1, help="The batch size for evaluation.")
    
    return parser


def metadataPresent(enginePath: str):
    """If we want to validate a TensorRT engine, check if there are already metadata present inside the .engine file.
    This is because the custom TensorRT exporter does not add metadata by default and Ultraytics' validator expects some metadata
    in order to execute.
    """
    
    enginePath = Path(enginePath)
    engineBytes = enginePath.read_bytes()

    # Check if metadata is already present
    try:
        metadataLength = int.from_bytes(engineBytes[:4], byteorder="little")
        existing = json.loads(engineBytes[4:4 + metadataLength].decode("utf-8"))
        print(f"Metadata already present: {list(existing.keys())}")
        return True
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("No metadata present.")
        return False


def embedMetadata(enginePath: str, imageSize: int, batchSize: int):
    """
    Prepend metadata to a TensorRT engine file in the format
    Ultralytics TensorRTBackend expects:

      [4-byte little-endian length][json bytes][engine bytes]

    Ultralytics reads the first 4 bytes as the JSON length,
    then reads that many bytes as JSON, then deserializes
    the rest as the TRT engine.
    """
    
    # Create a copy of the engine file so that we do not alter the original
    enginePath = Path(enginePath)
    engineCopyPath = str(enginePath.with_suffix('')) + "_copy" + enginePath.suffix
    shutil.copyfile(enginePath, engineCopyPath)
    engineCopyPath = Path(engineCopyPath)
    
    # Read existing engine bytes
    engine_bytes = engineCopyPath.read_bytes()

    metadata = {
        "task":   "detect",
        "batch":  batchSize,
        "imgsz":  imageSize,
        "stride": 32,
        "nc":     10,
        "names": {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor",
        }
    }
    # Encode metadata as JSON
    json_bytes = json.dumps(metadata).encode("utf-8")

    # 4-byte little-endian length prefix
    length_prefix = struct.pack("<I", len(json_bytes))

    # Write: [length][json][engine]
    engineCopyPath.write_bytes(length_prefix + json_bytes + engine_bytes)

    print(f"Metadata embedded: {len(json_bytes)} bytes prepended to {engineCopyPath}")
    print(f"Keys: {list(metadata.keys())}")    
    
    return str(engineCopyPath)


def main():

    parser = getParser()
    args = parser.parse_args()
    
    if args.modelPath.endswith(".engine") and not metadataPresent(args.modelPath):
        engineCopyPath = embedMetadata(args.modelPath, args.imageSize, args.batchSize)
        model = YOLO(engineCopyPath)
    else:
        engineCopyPath = "-----"
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

    if os.path.exists(engineCopyPath):
        os.remove(engineCopyPath)
        print(f"Removed {engineCopyPath}")

if __name__ == "__main__":
    main()

