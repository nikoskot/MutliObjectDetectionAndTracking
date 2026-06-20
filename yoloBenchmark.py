from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import argparse

def getParser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modelPath', required=True, help="The .pt model file to evaluate.")
    parser.add_argument("--imageSize", type=int, default=640, help="The image size that the model uses internally.")
    parser.add_argument("--halfPrecision", action="store_true", help="Export model in FP16 or not.")
    parser.add_argument("--int8Quantization", action="store_true", help="Export model in INT8 or not.")
    
    return parser


def main():
    
    parser = getParser()
    args = parser.parse_args()
    
    # Benchmark on GPU
    benchmark(model=args.modelPath, data="data.yaml", imgsz=args.imageSize, half=False, int8=False, device=0, verbose=True, format="-")
    benchmark(model=args.modelPath, data="data.yaml", imgsz=args.imageSize, half=True, int8=False, device=0, verbose=True, format="-")
    benchmark(model=args.modelPath, data="data.yaml", imgsz=args.imageSize, half=False, int8=True, device=0, verbose=True, format="-")
    benchmark(model=args.modelPath, data="data.yaml", imgsz=args.imageSize, half=False, int8=False, device=0, verbose=True, format="engine")
    benchmark(model=args.modelPath, data="data.yaml", imgsz=args.imageSize, half=True, int8=False, device=0, verbose=True, format="engine")
    benchmark(model=args.modelPath, data="data.yaml", imgsz=args.imageSize, half=False, int8=True, device=0, verbose=True, format="engine")

if __name__ == "__main__":
    main()