import cv2
import numpy as np

# ATTEMPT TEAM CLASSIFICATION USING WHITE/BLACK PIXEL RATIO
# NOT VERY SUCCESSFULL WOULD NEED TO RUN AN AVERAGE WITH THE TRACKLET TO PREVENT MISSCLAFICATION
 
# Read image
img = cv2.imread('images/basketballA1.png')

# Read COCO class names
with open('yolov4/coco.names', 'r') as f:
    classes = f.read().splitlines()

# Read yolov4 network configuration + weights 
net = cv2.dnn.readNetFromDarknet('yolov4/yolov4.cfg', 'yolov4/yolov4.weights')

# Initialise detection model 
model = cv2.dnn_DetectionModel(net)

# Set input parameters:
# 1/255 scale factor defines that pixel values will be scaled from 0 to 1
# Given image will be resized to 416x416 without cropping
# swapRB parameter defines that the first and last channels will be swapped because OpenCV uses BGR (and not RGB)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

# We detect objects in an image:
# Confidence score threshold used to filter boxes by confidence score set to 0.6
# Most appropriate boxes selected using non-maximum suppression (NMS), NMS is controlled by threshold and set to 0.4 
classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

# FIRST extract detection into a new image
for (classId, score, box) in zip(classIds, scores, boxes):
    # Only if the object's classId == [x]
    # [0], person
    # [43], knife
    # [49], orange
    if classId == [0]:
        # Extract detection into a new image
        extract = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] # img[y1:y2, x1:x2]

        # Convert from RGB to HSV
        img_hsv = cv2.cvtColor(extract, cv2.COLOR_RGB2HSV)

        # Display the image 
        cv2.imshow('Image', img_hsv)
        cv2.waitKey(0)

        MINCOLOR = np.array([0, 0, 0])
        MAXCOLOR = np.array([255, 255, 100])
        mask = cv2.inRange(img_hsv, MINCOLOR, MAXCOLOR)

        # Display the image 
        cv2.imshow('Image', mask)
        cv2.waitKey(0)

        res = cv2.bitwise_and(extract, extract, mask=mask)

        total_pix = extract.any(axis=-1).sum()
        color_pix = res.any(axis=-1).sum()
        ratio = color_pix/total_pix

        print(ratio)