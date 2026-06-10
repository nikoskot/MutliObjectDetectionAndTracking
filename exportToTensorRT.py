import argparse
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
from pathlib import Path
import random


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def getParser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--onnx', required=True, help="The .onnx file to use for export.")
    parser.add_argument("--imageSize", type=int, default=640, help="The image size that the model uses internally.")
    parser.add_argument("--fp16", action="store_true", help="Export model in FP16 or not.")
    parser.add_argument("--int8", action="store_true", help="Export model in INT8 or not.")
    parser.add_argument("--batchSize", type=int, default=1, help="The batch size to optimize the engine for.")
    parser.add_argument("--calibrationImagesDir", type=str, help="Path to calibration images folder")
    
    return parser


class VisDroneCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Feeds calibration images through the network so TRT can measure
    the activation ranges needed to set INT8 scale factors per layer.

    IInt8EntropyCalibrator2 uses entropy minimization to find optimal quantization thresholds,
    which works better than the simpler MinMax calibrator for most CV models.
    """

    def __init__(self, imageDir: str, batchSize: int = 1, imgsz: int = 640, cacheFile: str = "calibration.cache", maxImages: int = 500):

        super().__init__()

        self.batchSize = batchSize
        self.imgsz      = imgsz
        self.cacheFile = cacheFile

        # Collect image paths. More images = better calibration, but longer build time.
        paths = list(Path(imageDir).glob("*.jpg"))
        random.shuffle(paths)
        self.imagePaths = paths[:maxImages]
        self.index = 0

        # Allocate page-locked host memory and GPU memory for one batch. Page-locked (pinned) memory allows faster H2D transfers via DMA
        self.batchBytes = batchSize * 3 * imgsz * imgsz * np.dtype(np.float32).itemsize
        self.deviceInput = cuda.mem_alloc(self.batchBytes)  # allocate memory space in GPU VRAM
        self.hostBuffer  = cuda.pagelocked_empty(batchSize * 3 * imgsz * imgsz, np.float32) # allocate pagelocked memory space in CPU RAM

    def get_batch_size(self) -> int:
        return self.batchSize

    def get_batch(self):
        """
        Called repeatedly by TRT during engine build.
        Must return a list of device pointers (one per input tensor),
        or None when all calibration images are exhausted.

        TRT calls this until it returns None, then finalizes scale factors.
        """

        if self.index >= len(self.imagePaths):
            return None  # signals TRT that calibration is complete

        batchImages = []
        for _ in range(self.batchSize):
            if self.index >= len(self.imagePaths):
                # Pad last batch by repeating the last image
                imgPath = self.imagePaths[-1]
            else:
                imgPath = self.imagePaths[self.index]
                self.index += 1

            # Must use IDENTICAL preprocessing to the inference pipeline
            # Any mismatch here means wrong scale factors and accuracy loss
            img = cv.imread(str(imgPath))
            img = self._preprocess(img)
            batchImages.append(img)

        batch = np.stack(batchImages, axis=0).astype(np.float32)

        np.copyto(self.hostBuffer, batch.ravel())   # This is a CPU-to-CPU copy. It moves the preprocessed batch (a regular numpy array) into the pinned/pagelocked buffer.
        cuda.memcpy_htod(self.deviceInput, self.hostBuffer) # This is the actual H2D (Host to Device) transfer. The DMA engine reads from the pinned buffer and writes directly into GPU VRAM.

        return [int(self.deviceInput)]

    def read_calibration_cache(self):
        """
        If a cache file exists, TRT loads scale factors from it
        instead of running calibration again. This lets us build
        multiple INT8 engines from one calibration run.
        """
        if Path(self.cacheFile).exists():
            print(f"Loading calibration cache: {self.cacheFile}")
            with open(self.cacheFile, "rb") as f:
                return f.read()
        return None  # no cache — TRT will run full calibration

    def write_calibration_cache(self, cache):
        """Called by TRT after calibration. Save calibration cache for future builds."""
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print(f"Calibration cache saved: {self.cacheFile}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Must exactly match the inference pipeline preprocessing.
        """
        h, w  = frame.shape[:2]
        scale = self.imgsz / max(h, w)
        newH, newW = int(h * scale), int(w * scale)

        resized = cv.resize(frame, (newW, newH), interpolation=cv.INTER_LINEAR)

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        # Calculate centered padding offsets
        padTop  = (self.imgsz - newH) // 2
        padLeft = (self.imgsz - newW) // 2

        canvas[padTop:padTop + newH, padLeft:padLeft + newW] = resized

        canvas = canvas[:, :, ::-1]                          # BGR → RGB
        canvas = canvas.transpose(2, 0, 1)                   # HWC → CHW
        canvas = np.ascontiguousarray(canvas)
        canvas = canvas.astype(np.float32) / 255.0           # normalize

        return canvas


def build_engine(onnxPath: str, enginePath: str, fp16: bool = True, int8: bool = True, imageDir: str = None, cacheFile: str = "calibration.cache", batchSize: int = 1, imageSize: int = 640):

    if int8 and imageDir is None:
        raise ValueError("imageDir is required for INT8 calibration")
    
    builder = trt.Builder(TRT_LOGGER)

    # The network holds the parsed graph
    network = builder.create_network()

    # Parse the ONNX file into the TRT network
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnxPath, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    # BuilderConfig controls optimization strategy
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))   # 4 GB workspace

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    elif int8:
        # INT8 requires both flags. TRT uses FP16 for layers where INT8 would cause too much accuracy loss (automatic fallback)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)

        calibrator = VisDroneCalibrator(
            imageDir  = imageDir,
            batchSize = batchSize,
            imgsz     = imageSize,
            cacheFile = cacheFile,
        )
        config.int8_calibrator = calibrator

    # Build and serialize
    print("Building engine. This takes a few minutes...")
    serialized = builder.build_serialized_network(network, config)

    # Save engine to disk
    with open(enginePath, "wb") as f:
        f.write(serialized)
    print(f"Engine saved at {enginePath}.")


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def main():
    
    parser = getParser()
    args = parser.parse_args()
    
    outputName = f"{''.join(args.onnx.split('.')[:-1])}{'_fp16' if args.fp16 else ''}{'_int8' if args.int8 else ''}.engine"

    # Build the TensorRT engine and save it
    build_engine(onnxPath=args.onnx, enginePath=outputName, fp16=args.fp16, int8=args.int8, imageDir=args.calibrationImagesDir, cacheFile="int8Calibration.cache", batchSize=args.batchSize, imageSize=args.imageSize)
    
    # Validate output size
    engine = load_engine(outputName)
    context = engine.create_execution_context()
    # context.set_input_shape("images", (args.batchSize, 3, args.imageSize, args.imageSize))

    dummy = np.zeros((args.batchSize, 3, args.imageSize, args.imageSize), dtype=np.float32)

    # Alloc GPU buffers
    dInput  = cuda.mem_alloc(dummy.nbytes)
    dOutput = cuda.mem_alloc(args.batchSize * 14 * 8400 * 4)  # float32 - 14 instead of 84 because the model is trained on 10 classes

    cuda.memcpy_htod(dInput, dummy)
    context.execute_v2([int(dInput), int(dOutput)])

    output = np.empty((args.batchSize, 14, 8400), dtype=np.float32)
    cuda.memcpy_dtoh(output, dOutput)

    print("TRT output shape:", output.shape)
    print("Max value:", output.max())
    

if __name__ == "__main__":
    main()