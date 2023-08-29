import os
import cv2
 
# STEP 1: Open video file
capture = cv2.VideoCapture('videos/video.mp4')

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

# STEP 3: Using each frames, recreate video
sample = cv2.imread(images_paths[0])
height, width, layers = sample.shape
video = cv2.VideoWriter('videos/new_video.mp4', -1, 30, (width, height)) # video output file name, codec, number of frames per second, widht & heigth

for image_path in images_paths:
    image = cv2.imread(image_path)
    video.write(image)

cv2.destroyAllWindows()
video.release()

# STEP 4: After the new video is created, delete all the individual frames
for image_path in images_paths:
    os.remove(image_path)