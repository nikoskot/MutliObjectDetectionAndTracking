import configargparse
import cv2 as cv
import time
from ultralytics import YOLO
from tracker.byte_tracker import BYTETracker
import numpy as np


def getParser():
    parser = configargparse.ArgParser(default_config_files=["config.yaml"])
    parser.add_argument("--configFile", is_config_file=True, help="Config file path")
    parser.add_argument("--videoSource", type=str, help="Path to video file or image sequence")
    parser.add_argument("--modelPath", type=str, help="Path to YOLO model file .pt or .engine")
    parser.add_argument("--trackThresh", type=float, default=0.5, help="Tracking confidence threshold")
    parser.add_argument("--trackBuffer", type=int, default=30, help="Number of frames to keep lost tracks")
    parser.add_argument("--matchThresh", type=float, default=0.8, help="Matching threshold for tracker")
    
    return parser


def main():
    parser = getParser()
    args = parser.parse_args()
    
    # Setup input source
    if args.videoSource.endswith(".jpg") or args.videoSource.endswith(".png"):
        cap = cv.VideoCapture(args.videoSource, cv.CAP_IMAGES)
    elif args.videoSource.endswith(".mp4") or args.videoSource.endswith(".avi"):
        cap = cv.VideoCapture(args.videoSource, cv.CAP_ANY)
    
    if not cap.isOpened():
        print(f"Cannot open video source {args.videoSource}.")
        return
    else: 
        print("Video source opened.")
    
    # Setup model
    inferenceModel = YOLO(model=args.modelPath, task="detect", verbose=True)
    
    # Setup tracker
    tracker = BYTETracker(track_thresh=args.trackThresh, track_buffer=args.trackBuffer, match_thresh=args.matchThresh)
        
    totalFrames = 0
    start = time.perf_counter()
    
    while True:
        
        # Get frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from video source (stream end?). Exiting ...")
            break
        totalFrames += 1
        
        cv.imshow("Original input", frame)
        
        # Detection
        results = inferenceModel(source=frame, device=0, verbose=False)
        res = results[0].cpu()
        
        # Visualize detection results
        frameCopy = frame.copy()
        for x1, y1, x2, y2, conf, cls in res.boxes.data:
            cv.rectangle(frameCopy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv.putText(frameCopy, f"{res.names[int(cls)]} {conf:.2f}", (int(x1), int(y1) - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imshow("Detection results", frameCopy)
        
        # Tracking
        scale = min(inferenceModel.imgsz / float(res.orig_shape[0]), inferenceModel.imgsz / float(res.orig_shape[1]))
        boxes_model_space = res.boxes.xyxy * scale
        detectionsForTracker = np.concatenate((boxes_model_space, res.boxes.conf[:, np.newaxis], res.boxes.cls[:, np.newaxis]), axis=1)
        onlineTargets = tracker.update(detectionsForTracker, res.orig_shape, (inferenceModel.imgsz, inferenceModel.imgsz))
        
        # Visualize tracking results
        frameCopy = frame.copy()
        for t in onlineTargets:
            tlbr = t.tlbr
            tid = t.track_id
            classId = t.class_id
            cv.rectangle(frameCopy, (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3])), (0, 255, 0), 1)
            cv.putText(frameCopy, f"{res.names[classId]}_{tid}", (int(tlbr[0]), int(tlbr[1]) - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imshow("Tracking results", frameCopy)
        
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