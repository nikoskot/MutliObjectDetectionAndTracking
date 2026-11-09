import argparse

import yaml

from capture import create_capturer
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
    
    # Setup logger
    logger = Logger(config["logs_dir"])
    
    # Setup monitor
    monitor = Monitor(logger)
    
    capturer_config = config.get("capturer", {})
    
    with create_capturer(capturer_config, logger, monitor) as capturer:
    
        i = 0
        while i < 1000:
            monitor.reset_durations()
            
            frame = capturer.read()
            if frame is None:
                break
        
            monitor.log_stages_durations()
    

if __name__ == "__main__":
    main()