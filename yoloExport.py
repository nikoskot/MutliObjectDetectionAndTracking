from ultralytics import YOLO
import argparse

def getParser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modelPath', required=True, help="The .pt model file to evaluate.")
    parser.add_argument("--outputFormat", default="engine", choices=["engine", "onnx"], help="The output format.")
    parser.add_argument("--imageSize", type=int, default=640, help="The image size that the model uses internally.")
    parser.add_argument("--halfPrecision", action="store_true", help="Export model in FP16 or not.")
    parser.add_argument("--int8Quantization", action="store_true", help="Export model in INT8 or not.")
    
    return parser


def main():
    
    parser = getParser()
    args = parser.parse_args()
    
    model = YOLO(args.modelPath)
    
    # Export model to TensorRT engine
    model.export(format=args.outputFormat, imgsz=args.imageSize, half=args.halfPrecision, int8=args.int8Quantization, device=0)


if __name__ == "__main__":
    main()