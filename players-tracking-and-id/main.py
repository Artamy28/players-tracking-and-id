import os
import cv2
import numpy as np

import courtdetection_sparse
import yolov4_deepsort.tracker

# Open video file
capture = cv2.VideoCapture('videos/best_trim.mp4')

# Initialise necessary variables
formatpoints = []
formatsourcepoints = []
frame_count = 0
frame = None

# Read video frame by frame
while(capture.isOpened()):
    old_frame = frame
    ret, frame = capture.read()
    # End loop when video ends
    if ret == False:
        break
    
    # Increment frame count
    frame_count += 1

    # STEP1: COURT DETECTION
    # If it's the first frame, we need to ask for user input
    if frame_count == 1:
        result, points, sourcepoints = courtdetection_sparse.firstdetection(frame)
        
        # Convert points into correct format for optical flow function
        for point in points[0]:
            formatpoints.append([point])
        #formatpoints = np.array(formatpoints, np.float32)

    # Otherwise compute court detection from the previous detection + optical flow
    else:
        result, points, sourcepoints = courtdetection_sparse.nextdetection(old_frame, frame, formatpoints, sourcepoints)

    # STEP2: PLAYER DETECTION AND TRACKING
    yolov4_deepsort.detect_and_track(capture, frame)

    # Display the frame
    # cv2.imshow('Frame', result)
    # Press 'Q' to exit, wait _ milliseconds between each frames
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
# Release video capture object
capture.release()
# Close all windows
cv2.destroyAllWindows()