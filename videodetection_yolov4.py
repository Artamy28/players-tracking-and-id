import os
import cv2
 
# STEP 1: Open video file
capture = cv2.VideoCapture('videos/best.mp4')

# STEP 2: Read video frame by frame, save each frame into a list
i=0
images_paths = []
while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == False:
        break
    
    cv2.imwrite('frames/frame' + str(i) + '.jpg', frame)
    images_paths.append('frames/frame' + str(i) + '.jpg')
    i+=1

capture.release()
cv2.destroyAllWindows

# Apply object detection to each frame in the list
i=0
editedimages_paths = []
for image_path in images_paths:
    # Read image
    img = cv2.imread(image_path)

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

    # THEN draw bounding boxes along with class labels and scores on an image
    for (classId, score, box) in zip(classIds, scores, boxes):
        # Only if the object's classId == [x]
        # [0], person
        # [43], knife
        # [49], orange
        if classId == [0]:
            # Draw bounding boxes
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), # img, (x1, y1), (x2, y2)
                        color=(0, 255, 0), thickness=2)

            # Draw class labels and scores
            text = '%s: %.2f' % (classes[classId[0]], score)
            cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(0, 255, 0), thickness=2)

    # Save the image 
    cv2.imwrite('editedframes/frame' + str(i) + '.jpg', img)
    editedimages_paths.append('editedframes/frame' + str(i) + '.jpg')
    i+=1

# STEP 3: Using each frames, recreate video
sample = cv2.imread(editedimages_paths[0])
height, width, layers = sample.shape
video = cv2.VideoWriter('videos/new_video2.mp4', -1, 30, (width, height)) # video output file name, codec, number of frames per second, widht & heigth

for editedimage_path in editedimages_paths:
    image = cv2.imread(editedimage_path)
    video.write(image)

cv2.destroyAllWindows()
video.release()

"""
# STEP 4: After the new video is created, delete all the individual frames
for image_path in images_paths:
    os.remove(image_path)

for editedimage_path in editedimages_paths:
    os.remove(editedimage_path)
"""