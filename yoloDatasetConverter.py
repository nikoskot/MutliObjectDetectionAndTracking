"""This script converts the VisDrone2019-MOT train/val/test datasets to a format the YOLO from Ultralytics expects."""

from pathlib import Path
import configargparse
import os
import shutil
import cv2 as cv
from tqdm import tqdm


VISDRONE_CATEGORIES_TO_KEEP = [1, 4, 5, 6, 9]
CATEGORIES_MAPPING = {1:0, 4:1, 5:2, 6:3, 9:4}


def getParser():
    parser = configargparse.ArgParser(default_config_files=[".\yoloDatasetConverterConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help="Config file path")
    parser.add("--split", choices=['train', 'val'], help="The dataset split to convert")
    parser.add("--sequencesPath", type=lambda p: Path(p).resolve(), help="The path of the folder that contains the sequences")
    parser.add("--annotationsPath", type=lambda p: Path(p).resolve(), help="The path of the folder that contains the annotations")
    parser.add("--resultsPath", type=lambda p: Path(p).resolve(), help="The path of the resulting dataset")
    return parser


def main():
    parser = getParser()
    args = parser.parse_args()
    
    # images folder -> 
    # for each sequence -> get all available images with the paths that contain the name of the sequense and the image name (which is the frame numebr)
    # create an empty list with size the same as the number of images in the sequence. Each element of this list will contain the annotations for a specific image (the text of the corresponding resulting annotation txt file)
    # for each image in the sequence-> copy the image to the destination folder with the name 'sequenceName_imageName.jpg'
    # read the annotations txt of the sequence
    # for each line -> get the frame number from the first part of the line string, convert the bbox description to YOLO format, get the class id and convert it to YOLO format,
    # create a string with the annotation info, append this string to the end of the corresponding element of the annotations list (based on the frame number)
    # write the content of the annotations list to the annotations txt files, one file for each element
    # move to the next sequence
    
    # Create resulting dataset folders
    os.makedirs(os.path.join(args.resultsPath, args.split, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.resultsPath, args.split, "labels"), exist_ok=True)
    
    # Get all sequences folders
    sequences = [d for d in os.listdir(args.sequencesPath) if os.path.isdir(os.path.join(args.sequencesPath, d))]
    
    for seq in tqdm(sequences):

        # Get all the names of all the frames in the current sequence
        frames = sorted([fr for fr in os.listdir(os.path.join(args.sequencesPath, seq)) if fr.endswith('.jpg')])
        imageHeight, imageWidth = cv.imread(os.path.join(args.sequencesPath, seq, frames[0])).shape[:2]
        
        # Create an empty list with size the same as the number of images in the current sequence
        # Each element of this list will contain the annotations for a specific image (the text of the corresponding resulting annotation txt file)
        annotationsStrings = [''] * len(frames)
        
        # Copy the image to the destination folder with the name 'sequenceName_imageName.jpg'
        for fr in frames:
            shutil.copy(os.path.join(args.sequencesPath, seq, fr), os.path.join(args.resultsPath, args.split, "images", f"{seq}_{fr}"))
            
        # Read the annotations txt file of the sequence
        with open(os.path.join(args.annotationsPath, f"{seq}.txt"), "r") as f:
            lines = f.readlines()
            
            for l in lines:
                l = l.split(',')
                
                # Ignore line if the detection is not MOT related category, or if score is 0
                if (int(l[7]) not in VISDRONE_CATEGORIES_TO_KEEP) or (int(l[6]) == 0):
                    continue
                
                # frameNumberString = l[0].zfill(7)
                newClassId = CATEGORIES_MAPPING[int(l[7])]
                cx = (int(l[2]) + int(l[4]) / 2) / imageWidth
                cy = (int(l[3]) + int(l[5]) / 2) / imageHeight
                w  = int(l[4]) / imageWidth
                h  = int(l[5]) / imageHeight
                
                annotationsStrings[int(l[0])-1] += f"{newClassId} {cx} {cy} {w} {h}\n"
        
        # Write the content of the annotations list to the annotations txt files, one file for each element
        for i, annotation in enumerate(annotationsStrings):
            with open(os.path.join(args.resultsPath, args.split, "labels", f"{seq}_{str(i+1).zfill(7)}.txt"), "w") as f:
                f.write(annotation)


if __name__ == "__main__":
    main()