import cv2 as cv
import numpy as np
import os
from itertools import groupby
from pathlib import Path


def main():
    numClasses = 4
    classesNames = ['pedestrian', 'car', 'van', 'truck', 'bus']
    classesColors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
    datasetPath = 'D:\\Datasets\\VisDrone2019_MOT_YOLO\\val'
    # imageHeight, imageWidth = 640, 640

    imagesFiles = os.listdir(os.path.join(datasetPath, 'images'))
    annotationsFiles = os.listdir(os.path.join(datasetPath, 'labels'))

    imagesFilesBySequence = [list(j) for i, j in groupby(imagesFiles, lambda x:x[:10])]

    for sequence in imagesFilesBySequence:
        for i, framePath in enumerate(sequence):
            
            # Read frame
            frame = cv.imread(os.path.join(datasetPath, 'images', framePath), cv.IMREAD_COLOR)
            imageHeight, imageWidth = frame.shape[:2]
            
            # Read bbox annotations
            annotations = readAnnotations(os.path.join(datasetPath, 'labels', Path(framePath).stem + '.txt'))
            # Draw annotaion on frame
            for ann in annotations:
                unnormalizedAnnotation = [int(ann[0]), int(ann[1]*imageWidth), int(ann[2]*imageHeight), int(ann[3]*imageWidth), int(ann[4]*imageHeight)]
                boundingBoxUpperLeft = (unnormalizedAnnotation[1] - (unnormalizedAnnotation[3] // 2), unnormalizedAnnotation[2] - (unnormalizedAnnotation[4] // 2))
                boundingBoxLowerRight = (unnormalizedAnnotation[1] + (unnormalizedAnnotation[3] // 2), unnormalizedAnnotation[2] + (unnormalizedAnnotation[4] // 2))
                frame = cv.rectangle(frame, boundingBoxUpperLeft, boundingBoxLowerRight, classesColors[unnormalizedAnnotation[0]])
            
            cv.imshow(framePath[:10], frame)
            key = cv.waitKey(100) & 0xFF
            
            if key == ord('q'):  # q to quit
                cv.destroyAllWindows()
                return

            elif key == 32:  # SPACE key (ASCII 32) move to next sequence
                break
        
        cv.destroyAllWindows()
        
def readAnnotations(filePath):
    annotations = []
    with open(filePath) as f:
        lines = f.readlines()
        for l in lines:
            annotations.append([float(x) for x in l.split()])
    
    return annotations
            
    
if __name__ == '__main__':
    main()