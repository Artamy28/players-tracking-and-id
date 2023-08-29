import os
import cv2

# INPUT: VIDEO FILE
# OUTPUT: INDIVIDUAL IMAGES OF EACH PERSON DETECTION
# Create data to be used for team classification
 
# STEP 1: Open video file
capture = cv2.VideoCapture('videos/best.mp4')
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
print("Video FPS: ", fps)
print("Number of Frames: ", frame_count)
print("Video Duration: ", duration)

# STEP 2: Read video frame by frame
i=0
framecount = 0
while(capture.isOpened()):
    # Read image
    ret, frame = capture.read()
    if ret == False:
        break
    img = frame
    framecount += 1

    # Gather data every 60 frames (if the video is 30fps, we gather 1 frame per 2 seconds) - This number is customizable
    if framecount % 60 == 0:
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
                #cv2.imshow('Image', extract)
                #cv2.waitKey(0)
                cv2.imwrite('dataTC/img' + str(i) + '.jpg', extract)
                i+=1

cv2.destroyAllWindows()
capture.release()