import argparse
import numpy as np

import yaml

from capture import create_capturer
from detect import create_detector
from tracker.byte_tracker import BYTETracker
from logger import Logger
from monitor import Monitor


def getParser():
    parser = argparse.ArgumentParser(description="Argument parser for baseline detection pipeline")
    parser.add_argument("cfg", type=str, default="config.yaml", nargs="?", help="Config file path")
    
    return parser

def main():
    # Get config arguments
    args = getParser().parse_args()
    with open(args.cfg, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    capturer_config = config.get("capturer", {})
    class_taxonomy = config.get("class_taxonomy", {})
    detector_config = config.get("detector", {})
    
    # Setup logger
    logger = Logger(config["logs_dir"])
    logger.info("MultiObjectDetectionAndTracking: Pipeline started.")
    
    # Setup monitor
    monitor = Monitor(logger)
    
    # Setup detector
    detector = create_detector(detector_config, class_taxonomy, logger, monitor)
    detector.load()
    # detector.warmup()
    
    # Setup tracker
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30, logger=logger, monitor=monitor)
    
    with create_capturer(capturer_config, logger, monitor) as capturer:
    
        i = 0
        while i < 1000:
            monitor.reset_durations()
            
            # Capture frame
            frame = capturer.read()
            if frame is None:
                break
            
            # Detect
            detections = detector.detect(frame)
            
            # Track
            detectionsForTracker = np.concatenate((detections.boxes, detections.scores[:, np.newaxis], detections.class_ids[:, np.newaxis]), axis=1)
            onlineTargets = tracker.track(detectionsForTracker)
            
            monitor.log_stages_durations()
    
    detector.release()
    
    
if __name__ == "__main__":
    main()