import configargparse
import cv2 as cv
import time
from ultralytics import YOLO


def getParser():
    parser = configargparse.ArgParser(default_config_files=["config.yaml"])
    parser.add_argument("--configFile", is_config_file=True, help="Config file path")
    # parser.add_argument("")
    
    return parser


def main():
    parser = getParser()
    args = parser.parse_args()
    
    # Setup input source
    videoSource = "D:\\Datasets\\VisDrone2019-MOT-test-dev\\sequences\\uav0000009_03358_v\\%07d.jpg"
    # videoSource = "D:\\Downloads\\65495-514501835_tiny.mp4"
    
    cap = cv.VideoCapture(videoSource, cv.CAP_IMAGES)
    # cap = cv.VideoCapture(videoSource, cv.CAP_FFMPEG)
    # cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"Cannot open video source {videoSource}.")
        return
    else: 
        print("Video source opened.")
    
    # Setup model
    inferenceModel = YOLO(model="best.engine", task="detect", verbose=True)
        
    totalFrames = 0
    start = time.perf_counter()
    
    while True:
        
        # Get frames
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from video source (stream end?). Exiting ...")
            break
        totalFrames += 1
        
        cv.imshow("Original input", frame)
        results = inferenceModel(source=frame, device=0, verbose=False)
        # res = next(results)
        res = results[0].cpu()
        # print(res.boxes.data)

        # cv.imshow("Annotated results", next(results).plot())
        # cv.imshow("Annotated results", results[0].plot())
        
        for x1, y1, x2, y2, conf, cls in res.boxes.data:
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv.putText(frame, f"{res.names[int(cls)]} {conf:.2f}", (int(x1), int(y1) - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imshow("Annotated results", frame)
        
        
        key = cv.waitKey(1) & 0xFF
        # Quit on 'q'
        if key == ord('q'):  
            break
        
    cap.release()
    cv.destroyAllWindows()
    
    elapsed = time.perf_counter() - start
    print(f"Processed {totalFrames} in {elapsed} seconds | {totalFrames / elapsed} fps average")
    
    # for res in allResults:
    #     for r in res:
    #         print(r.boxes)


if __name__ == "__main__":
    main()