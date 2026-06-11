import torch
from ultralytics import YOLO
import argparse
import onnx
import onnxruntime as ort
import numpy as np

def getParser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help="The .pt model file to use for export.")
    parser.add_argument("--imageSize", type=int, default=640, help="The image size that the model uses internally.")
    parser.add_argument("--batchSize", type=int, default=1, help="The batch size to optimize the model for.")
    
    return parser


def main():
    
    parser = getParser()
    args = parser.parse_args()
    
    # Load the model but extract the underlying nn.Module
    model = YOLO(args.model)
    torchModel = model.model
    torchModel.eval()

    device = torch.device("cuda")
    torchModel = torchModel.to(device)
    
    # Dummy input, defines the input shape baked into the ONNX graph
    dummyInput = torch.zeros(1, 3, args.imageSize, args.imageSize, device=device)
    
    outputName = f"{''.join(args.model.split('.')[:-1])}_bs{args.batchSize}.onnx"
    
    onnxProgram = torch.onnx.export(
        torchModel,
        dummyInput,
        None,
        opset_version=None,
        dynamo=True,
        optimize=True,
        input_names=["images"],
        output_names=["output0"],
        do_constant_folding=True,
    )
    onnxProgram.save(outputName)

    print(f"ONNX export done. ONNX format saved as {outputName}. \nVerifying ONNX graph...")
    
    # Check the graph structure
    modelONNX = onnx.load(outputName)
    onnx.checker.check_model(modelONNX)
    
    # Remove redundant outputs from the graph
    finalOutput = modelONNX.graph.output[0]
    del modelONNX.graph.output[:]
    modelONNX.graph.output.append(finalOutput)
    onnx.checker.check_model(modelONNX)
    onnx.save(modelONNX, outputName)
    print("ONNX graph valid.")

    # Run inference through ONNX Runtime to verify outputs match PyTorch
    session = ort.InferenceSession(outputName, providers=["CUDAExecutionProvider"])
    dummyInput = np.zeros((1, 3, args.imageSize, args.imageSize), dtype=np.float32)
    outputs = session.run(None, {"images": dummyInput})
    print("ONNX output shape:", outputs[0].shape)
    print("ONNX outputs:", len(outputs))


if __name__ == "__main__":
    main()