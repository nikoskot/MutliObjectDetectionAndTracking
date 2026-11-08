import argparse
import cv2 as cv
import yaml
from ultralytics import YOLO
from tracker.byte_tracker import BYTETracker
import numpy as np
from collections import deque


def getParser():
    parser = argparse.ArgumentParser(description="Argument parser for baseline detection pipeline")
    parser.add_argument("--cfg", type=str, default="config.yaml", help="Config file path")
    
    return parser

def main():
    args = getParser().parse_args()
    config = yaml.safe_load(args.cfg)
    

if __name__ == "__main__":
    main()