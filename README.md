# Multi Object Detection and Tracking

## Project Overview

This portfolio project explores object detection and tracking on the VisDrone dataset using YOLOv8 and TensorRT.

The core idea is:
- Use a YOLO model for object detection on VisDrone images.
- Convert the trained model into ONNX and TensorRT engines.
- Compare inference performance across PyTorch, TensorRT FP32, FP16, and INT8.
- Create multi-object tracking pipeline using ByteTrack.

> Training a model from scratch was too slow on the local machine, so pretrained YOLOv8 models were downloaded instead.

## Dataset

The dataset configuration is defined in `data.yaml`.
This repo targets the VisDrone dataset classes:
- pedestrian
- people
- bicycle
- car
- van
- truck
- tricycle
- awning-tricycle
- bus
- motor

The dataset is converted to the Ultralytics YOLO format using `yoloDatasetConverter.py`.

## Models Used

The pretrained models were downloaded from Hugging Face:
- `mshamrai/yolov8m-visdrone`
- `mshamrai/yolov8s-visdrone`

These models are already trained on the 10 VisDrone classes and are used for evaluation and engine conversion.

## What Has Been Implemented

### 1. Dataset conversion

- `yoloDatasetConverter.py`
  - Converts VisDrone MOT data to YOLO format expected by Ultralytics.
  - Reads sequence images and annotations.
  - Copies images and writes YOLO-style label files.
  - Supports `train`, `val`, and `test_dev` splits.
- `yoloDatasetConverterConfig.yaml`
  - Stores conversion paths for dataset sequences, annotations, and output folders.

### 2. Dataset exploration

- `yoloDatasetExploration.py`
  - Visualizes dataset images with ground-truth YOLO annotations drawn on top.
  - Helps verify that conversion and labels are correct.

### 3. Model export

- `exportToONNX.py`
  - Loads a pretrained YOLOv8 model from `.pt`.
  - Exports the network to ONNX format.
  - Verifies the ONNX graph and runs a sanity-check inference with ONNX Runtime.

- `exportToTensorRT.py`
  - Builds TensorRT engines from ONNX.
  - Supports FP32, FP16, and INT8 export modes.
  - Includes an INT8 calibrator tailored for VisDrone preprocessing.
  - Validates the built engine by running a dummy inference pass.

### 4. Model evaluation

- `yoloEvaluate.py`
  - Evaluates `.pt` models and TensorRT `.engine` files on VisDrone validation split.
  - Uses Ultralytics validation APIs to compute AP metrics.
  - Optionally embeds metadata into TensorRT engine files so that Ultralytics can load them.
  - Outputs per-class AP tables and global mAP metrics.

### 5. Inference benchmarking

- `benchmark.py`
  - Benchmarks pure inference performance with dummy input.
  - Measures latency statistics (mean, P50, P95, P99, std) and throughput (FPS).
  - Compares PyTorch `.pt` model vs TensorRT engines.

## Usage Examples

Convert VisDrone data:
```bash
python yoloDatasetConverter.py --split val --sequencesPath "D:\Datasets\VisDrone2019-MOT-YOLO\val\images" --annotationsPath "D:\Datasets\VisDrone2019-MOT-YOLO\val\annotations" --resultsPath "D:\Datasets\VisDrone2019_MOT_YOLO"
```

Export a model to ONNX:
```bash
python exportToONNX.py -m yolov8m_visdrone.pt --imageSize 640 --batchSize 1
```

Build a TensorRT engine:
```bash
python exportToTensorRT.py -o yolov8m_visdrone_bs1.onnx --imageSize 640 --batchSize 1 --fp16
```

Evaluate a model or engine:
```bash
python yoloEvaluate.py -m yolov8m_visdrone.pt --dataFilePath data.yaml --split val --imageSize 640 --batchSize 1
```

Benchmark inference performance:
```bash
python benchmark.py --pt yolov8m_visdrone.pt --fp32 yolov8m_visdrone_bs1.engine --fp16 yolov8m_visdrone_bs1_fp16.engine --int8 yolov8m_visdrone_bs1_int8.engine --imgsz 640 --runs 1000
```

## Results Summary

### Detection evaluation (validation split)

| Model | Precision | mAP50 | mAP75 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8m `.pt` | fp32 | 0.4643 | 0.2128 | 0.2359 |
| YOLOv8m TensorRT FP32 | fp32 | 0.4665 | 0.2142 | 0.2377 |
| YOLOv8m TensorRT FP16 | fp16 | 0.4664 | 0.2144 | 0.2377 |
| YOLOv8m TensorRT INT8 | int8 | 0.3232 | 0.1319 | 0.1574 |
| YOLOv8s `.pt` | fp32 | 0.4694 | 0.2195 | 0.2402 |
| YOLOv8s TensorRT FP32 | fp32 | 0.4697 | 0.2203 | 0.2413 |
| YOLOv8s TensorRT FP16 | fp16 | 0.4697 | 0.2205 | 0.2411 |
| YOLOv8s TensorRT INT8 | int8 | 0.1142 | 0.0485 | 0.0553 |

### Inference benchmarking

| Model | Precision | Batch | Mean latency (ms) | P50 (ms) | P95 (ms) | P99 (ms) | FPS |
|---|---|---|---|---|---|---|---|
| YOLOv8m `.pt` | fp32 | 1 | 23.34 | 23.25 | 23.99 | 24.33 | 42.85 |
| YOLOv8m `.pt` | fp16 | 1 | 132.35 | 132.34 | 133.87 | 137.02 | 7.56 |
| YOLOv8m TensorRT FP32 | fp32 | 1 | 19.40 | 19.35 | 19.55 | 20.53 | 51.56 |
| YOLOv8m TensorRT FP16 | fp16 | 1 | 10.70 | 10.67 | 10.80 | 11.41 | 93.48 |
| YOLOv8m TensorRT INT8 | int8 | 1 | 7.26 | 7.21 | 7.71 | 7.89 | 137.66 |
| YOLOv8s `.pt` | fp32 | 1 | 10.75 | 10.74 | 10.81 | 10.90 | 93.00 |
| YOLOv8s `.pt` | fp16 | 1 | 48.58 | 48.58 | 48.82 | 48.95 | 20.58 |
| YOLOv8s TensorRT FP32 | fp32 | 1 | 8.68 | 8.67 | 8.75 | 8.96 | 115.20 |
| YOLOv8s TensorRT FP16 | fp16 | 1 | 5.15 | 5.15 | 5.19 | 5.22 | 194.07 |
| YOLOv8s TensorRT INT8 | int8 | 1 | 3.98 | 3.98 | 4.02 | 4.07 | 250.99 |

## Next Steps

This repo is building toward a complete detection + tracking pipeline.
Planned work:
- Build a serialized detection + ByteTrack tracking pipeline using the `.pt` model.
- Then create an optimized TensorRT-based pipeline.
- Add pipeline-level optimizations such as parallelized preprocessing, inference, and postprocessing.

## Notes

- This is a portfolio project focused on model conversion, validation, and inference benchmarking.
- The current implementation does not yet include the full multi-object tracking pipeline.
- INT8 calibration requires careful dataset preprocessing and may need refinement.

## Dependencies

The code uses the following packages and frameworks:
- Python 3.x
- ultralytics
- torch
- onnx
- onnxruntime
- tensorrt
- pycuda
- opencv-python
- numpy
- prettytable
- tqdm
- configargparse

For TensorRT export and engine benchmarking, a CUDA-enabled GPU is required.

