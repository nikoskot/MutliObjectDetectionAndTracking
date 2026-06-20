"""
TensorRT Inference Benchmark
=============================
Measures latency (ms) and throughput (fps) for
FP32, FP16, and INT8 TensorRT engines vs PyTorch baseline.
"""

import argparse
import time
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from prettytable import PrettyTable
from ultralytics import YOLO
# import torch.cuda.nvtx as nvtx  # for Nsight Systems annotation


@dataclass
class BenchmarkResult:
    name: str
    precision: str
    batchSize: int
    latencyMeanMs: float
    latencyP50Ms: float
    latencyP95Ms: float
    latencyP99Ms: float
    latencyStdMs: float
    throughputFps: float
    gpuMemMb: float


class TRTEngine:
    """
    Loads a manually-built TensorRT engine for a YOLOv8-family detection
    model and exposes an infer() API.s
    """
 
    def __init__(self, enginePath: str):
        self.enginePath = Path(enginePath)
 
        self._loadEngine()
        self._allocateBuffers()
 
 
    def _loadEngine(self):
        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
 
        engineBytes = self.enginePath.read_bytes()
 
        self.engine  = runtime.deserialize_cuda_engine(engineBytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {self.enginePath}")
 
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
 
        print(f"Loaded engine: {self.enginePath.name}")
 
 
    def _allocateBuffers(self):
        """
        Iterates and binds the I/O tensors the engine reports.
        """
        self.inputs = []
        self.outputs = []
 
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode  = self.engine.get_tensor_mode(name)
 
            # Replace any dynamic -1 dims with a concrete size (static engines
            # won't have any, but this keeps the wrapper safe either way)
            concreteShape = tuple(max(s, 1) for s in shape)
            size = int(np.prod(concreteShape))
 
            hostMem   = cuda.pagelocked_empty(size, dtype)
            deviceMem = cuda.mem_alloc(hostMem.nbytes)
 
            self.context.set_tensor_address(name, int(deviceMem))
 
            tensor = {
                "name":   name,
                "shape":  concreteShape,
                "dtype":  dtype,
                "host":   hostMem,
                "device": deviceMem,
            }
 
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor)
            else:
                self.outputs.append(tensor)
        
        assert len(self.inputs) == 1, f"Expected 1 input, got {len(self.inputs)}"
        assert len(self.outputs) == 1, f"Expected 1 output, got {len(self.outputs)}"

        self.inputName  = self.inputs[0]["name"]
        self.outputName = self.outputs[0]["name"]
        self.batchSize  = self.inputs[0]["shape"][0]

        print(f"Bound input  '{self.inputName}'  {self.inputs[0]['shape']}")
        print(f"Bound output '{self.outputName}' {self.outputs[0]['shape']}")
        print(f"Batch size (fixed): {self.batchSize}")
 
 
    def _getTensor(self, name: str) -> dict:
        for t in self.inputs + self.outputs:
            if t["name"] == name:
                return t
        raise KeyError(f"No tensor named '{name}' bound on this engine")

 
    def infer(self, inputTensor: np.ndarray) -> np.ndarray:
        """
        Runs one forward pass. Copies input H2D, executes, copies the
        main detection output D2H.
        """
        inputBuf = self._getTensor(self.inputName)
 
        np.copyto(inputBuf["host"], inputTensor.ravel())
        cuda.memcpy_htod_async(inputBuf["device"], inputBuf["host"], self.stream)
 
        self.context.execute_async_v3(self.stream.handle)
 
        outputBuf = self._getTensor(self.outputName)
        cuda.memcpy_dtoh_async(outputBuf["host"], outputBuf["device"], self.stream)
 
        self.stream.synchronize()
 
        return outputBuf["host"].reshape(outputBuf["shape"]).copy()


class PyTorchEngine:
    
    def __init__(self, pt_path: str, precision: str = "fp32"):
        self.model = YOLO(pt_path).model.eval().cuda()

        if precision == "fp16":
            self.model = self.model.half()

        self.precision = precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32


    def infer(self, inputData: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(inputData).to("cuda", dtype=self.dtype)
        with torch.no_grad():
            out = self.model(tensor)
        return out[0].cpu().numpy() if isinstance(out, (list, tuple)) else out.cpu().numpy()


def measure(engine, batchSize: int, imgsz: int, warmupRuns: int, timedRuns: int, name: str, precision: str) -> BenchmarkResult:
    """
    Runs warmup runs discarded iterations then timed runs measured iterations.
    Returns a BenchmarkResult with latency percentiles and throughput.
    """

    dummy = np.random.rand(batchSize, 3, imgsz, imgsz).astype(np.float32)

    # Warmup. First N runs trigger CUDA kernel compilation (JIT)
    # nvtx.range_push(f"warmup_{name}")
    for _ in range(warmupRuns):
        engine.infer(dummy)
    torch.cuda.synchronize()
    # nvtx.range_pop()

    # Timed runs
    latencies = []

    # nvtx.range_push(f"benchmark_{name}_bs{batchSize}")

    for i in range(timedRuns):
        # nvtx.range_push(f"iter_{i}")

        # Use CUDA events for GPU-side timing
        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent   = torch.cuda.Event(enable_timing=True)

        startEvent.record()
        engine.infer(dummy)
        endEvent.record()

        torch.cuda.synchronize()  # wait for GPU before reading event time
        latencies.append(startEvent.elapsed_time(endEvent))  # ms

        # nvtx.range_pop()

    # nvtx.range_pop()

    latencies = np.array(latencies)

    # GPU memory allocated at peak during this run
    gpuMemMb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()

    totalTimeSecs = latencies.sum() / 1000.0
    totalFrames = timedRuns * batchSize
    throughput = totalFrames / totalTimeSecs

    return BenchmarkResult(
        name=name,
        precision=precision,
        batchSize=batchSize,
        latencyMeanMs=float(latencies.mean()),
        latencyP50Ms=float(np.percentile(latencies, 50)),
        latencyP95Ms=float(np.percentile(latencies, 95)),
        latencyP99Ms=float(np.percentile(latencies, 99)),
        latencyStdMs=float(latencies.std()),
        throughputFps=float(throughput),
        gpuMemMb=float(gpuMemMb),
    )


def printTable(results: list[BenchmarkResult]):
    table = PrettyTable(["Engine", "Prec", "BS", "Mean(ms)", "P50(ms)", "P95(ms)", "P99(ms)", "Std(ms)", "FPS", "GPU(MB)"])

    for r in results:
        table.add_row([r.name, 
                       r.precision, 
                       r.batchSize, 
                       np.round(r.latencyMeanMs, 2), 
                       np.round(r.latencyP50Ms, 2), 
                       np.round(r.latencyP95Ms, 2), 
                       np.round(r.latencyP99Ms, 2), 
                       np.round(r.latencyStdMs, 2), 
                       np.round(r.throughputFps, 2), 
                       np.round(r.gpuMemMb, 2)])

    table.title = "Benchmark Results"
    print(table)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt",      type=str, help="Path to .pt PyTorch model")
    p.add_argument("--fp32",    type=str, help="Path to FP32 .engine file")
    p.add_argument("--fp16",    type=str, help="Path to FP16 .engine file")
    p.add_argument("--int8",    type=str, help="Path to INT8 .engine file")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--warmup",  type=int, default=50,  help="Warmup iterations (discarded)")
    p.add_argument("--runs",    type=int, default=1000, help="Timed iterations per config")
    p.add_argument("--batchSz", type=int, default=1, help="Batch size to benchmark")
    return p.parse_args()


def main():
    args = parse_args()
    results = []

    enginesToTest = []

    if args.pt:
        enginesToTest.append(("PyTorch-FP32", "fp32", "pt", args.pt))
        enginesToTest.append(("PyTorch-FP16", "fp16", "pt", args.pt))

    if args.fp32:
        enginesToTest.append(("TRT-FP32", "fp32", "trt", args.fp32))

    if args.fp16:
        enginesToTest.append(("TRT-FP16", "fp16", "trt", args.fp16))

    if args.int8:
        enginesToTest.append(("TRT-INT8", "int8", "trt", args.int8))

    if not enginesToTest:
        print("No engines provided. Pass at least one of --pt, --fp32, --fp16, --int8")
        return

    for name, precision, kind, path in enginesToTest:
        print(f"\nLoading {name} from {path} ...")

        if kind == "trt":
            engine = TRTEngine(path)
        else:
            engine = PyTorchEngine(path, precision)

        result = measure(
            engine=engine,
            batchSize=args.batchSz,
            imgsz=args.imgsz,
            warmupRuns=args.warmup,
            timedRuns=args.runs,
            name=path,
            precision=precision,
        )
        results.append(result)

    printTable(results)


if __name__ == "__main__":
    main()