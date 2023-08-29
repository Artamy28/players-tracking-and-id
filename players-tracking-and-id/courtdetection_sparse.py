import cv2
import numpy as np

point_number = 0
def firstdetection(frame):
    # 1) STORE COORDINATES OF POINTS CLICKED ON THE IMAGE IN AN ARRAY
    points = []
    
    # Function to get the coordinates of the point clicked
    def click_event(event, x, y, flags, params):
        global point_number
        if event == cv2.EVENT_LBUTTONDOWN:
            coord = [x, y]
            points.insert(point_number, coord)
            print(x, ' ', y)

            point_number += 1
            if point_number == 14:
                cv2.destroyAllWindows()

        if event == cv2.EVENT_RBUTTONDOWN:
            points.insert(point_number, None)
            print('None')

            point_number += 1
            if point_number == 14:
                cv2.destroyAllWindows()

    # Read image
    img = frame
    # Display the image
    cv2.imshow('Image', img)
    # Set mouse handler for the image, and call the click_event function
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)

    # points = [None, None, None, None, None, None, (74, 248), None, (1338, 162), (1535, 336), (982, 380), (1693, 472), (1081, 522), None]
    # points = [None, None, None, None, None, None, None, None, (1205, 267), (1441, 469), (785, 539), (1628, 626), (910, 708), None]
    # points = [None, None, None, None, None, None, None, None, [1206.5331, 265.87482], [1442.3096, 468.00372], [786.8362, 537.95514], [1636.6663, 631.4011 ], [911.3899, 706.9685], None]
    points = [None, None, None, None, None, None, None, None, (1367, 274), (1593, 471), (962, 527), (1773, 623), (1077, 689), None]

    # 2) CONSTRUCT HOMOGRAPHY
    img_source = cv2.imread('images/basketballcourt.png')
    source_points = [
        [0, 0], [0, 172], [172, 172], [0, 281], [172, 281], [0, 452], 
        [426, 0], [426, 452], 
        [851, 0], [851, 172], [680, 172], [851, 281], [680, 281], [851, 452]
        ]

    hm_dest = []
    hm_source = []
    for i in range(14):
        if points[i] != None:
            hm_dest.append(points[i])
            hm_source.append(source_points[i])

    hm_dest = np.array([hm_dest])
    hm_source = np.array([hm_source])
    h, status = cv2.findHomography(hm_source, hm_dest)
    img_out = cv2.warpPerspective(img_source, h, (img.shape[1],img.shape[0]))

    #cv2.imshow("Source Image", img_source)
    #cv2.imshow("Destination Image", img)
    #cv2.imshow("Warped Source Image", img_out)

    # Convert the warped image to HSV, the goal is to create a mask
    hsv = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)

    # Lower and upper value to detect the black color
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([255, 255, 0])

    # Creating the mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Inverting the mask
    mask = cv2.bitwise_not(mask)

    # Apply the mask on the image frame
    result = cv2.bitwise_and(img, img, mask=mask)

    #cv2.imshow("HSV", hsv)
    #cv2.imshow("mask", mask)
    cv2.imshow("Masked Frame", result)
    cv2.waitKey(0)
    return result, hm_dest, hm_source 

# Format for points: [[-, -]], [[-, -]], [[-, -]], [[-, -]], [[-, -], ...]
def nextdetection(prevframe, currentframe, points, sourcepoints):
    # 1) GET NEW COORDINATES OF POINTS THROUGH OPTICAL FLOW
    oldimg = prevframe
    old_gray = cv2.cvtColor(oldimg, cv2.COLOR_BGR2GRAY)
    newimg = currentframe
    frame_gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)

    # Final dimension for p0 are: (Number of points, 1, 2)
    p0 = points
    
    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)

    # 2) CONSTRUCT HOMOGRAPHY
    img_source = cv2.imread('images/basketballcourt.png')

    hm_dest = []
    # Shape p1 destination points into the correct format for homography
    for point in p1:
        hm_dest.append(point[0])

    hm_dest = np.array([hm_dest])
    hm_source = sourcepoints

    h, status = cv2.findHomography(hm_source, hm_dest)

    img = currentframe
    img_out = cv2.warpPerspective(img_source, h, (img.shape[1],img.shape[0]))

    #cv2.imshow("Source Image", img_source)
    #cv2.imshow("Destination Image", img)
    #cv2.imshow("Warped Source Image", img_out)

    # Convert the warped image to HSV, the goal is to create a mask
    hsv = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)

    # Lower and upper value to detect the black color
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([255, 255, 0])

    # Creating the mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Inverting the mask
    mask = cv2.bitwise_not(mask)

    # Apply the mask on the image frame
    result = cv2.bitwise_and(img, img, mask=mask)

    #cv2.imshow("HSV", hsv)
    #cv2.imshow("mask", mask)
    cv2.imshow("Masked Frame", result)
    cv2.waitKey(0)
    return result, hm_dest, hm_source




