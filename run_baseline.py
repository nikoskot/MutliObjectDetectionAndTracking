import argparse

import yaml

from capture import create_capturer
from detect import create_detector
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
    
    # Setup monitor
    monitor = Monitor(logger)
    
    # Setup detector
    detector = create_detector(detector_config, class_taxonomy, logger, monitor)
    detector.load()
    # detector.warmup()
    
    with create_capturer(capturer_config, logger, monitor) as capturer:
    
        i = 0
        while i < 1000:
            monitor.reset_durations()
            
            frame = capturer.read()
            if frame is None:
                break
            
            detector.detect(frame)
            
            monitor.log_stages_durations()
    
    detector.release()
    
    
if __name__ == "__main__":
    main()