import configargparse
import cv2 as cv
import time
from ultralytics import YOLO
from tracker.byte_tracker import BYTETracker
import numpy as np
from collections import deque


def getParser():
    parser = configargparse.ArgParser(default_config_files=["config.yaml"])
    parser.add_argument("--configFile", is_config_file=True, help="Config file path")
    parser.add_argument("--modelPath", type=str, help="Path to YOLO model file .pt or .engine")
    parser.add_argument("--modelImageSize", type=int, default=640, help="Image that the image is transformed to before passing through the model.")
    
    return parser


def main():
    parser = getParser()
    args = parser.parse_args()
    
    # Setup input source
    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1360)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 764)
    
    if not cap.isOpened():
        print(f"Cannot open camera.")
        return
    else: 
        print(f"Camera is open.")
    
    # Setup model
    inferenceModel = YOLO(model=args.modelPath, task="detect", verbose=True)
            
    totalFrames = 0
    start = time.perf_counter()
    prevTime = time.perf_counter()
    frameTimes = deque(maxlen=30)
    
    while True:
        
        # Get frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from video source (stream end?). Exiting ...")
            break
        totalFrames += 1
        
        # Detection
        results = inferenceModel(source=frame, device=0, verbose=False)
        res = results[0].cpu()
        
        # Visualize detection results
        # frameCopy = frame.copy()
        for x1, y1, x2, y2, conf, cls in res.boxes.data:
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv.putText(frame, f"{res.names[int(cls)]} {conf:.2f}", (int(x1), int(y1) - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        currTime = time.perf_counter()
        frameTimes.append(currTime - prevTime)
        prevTime = currTime
        cv.putText(frame, f"FPS: {(len(frameTimes) / sum(frameTimes)):.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv.imshow("Detection results", frame)
        
        key = cv.waitKey(1) & 0xFF
        # Quit on 'q'
        if key == ord('q'):  
            break
        
    cap.release()
    cv.destroyAllWindows()
    
    elapsed = time.perf_counter() - start
    print(f"Processed {totalFrames} in {elapsed} seconds | {totalFrames / elapsed} fps average")


if __name__ == "__main__":
    main()