import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
# From images, compute MSER and SIFT Binary Array + compute 10-bin color histogram for each of the three RGB channel (resulting in a 30-bin color histogram)

# Creating array of 100 sift descriptors + 30-bins for blackteam 1
B1data = []
directory = 'dataID/BlackTeam/1_BJ_Taylor'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/BlackTeam/1_BJ_Taylor/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/BlackTeam/1_BJ_Taylor/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    B1data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for blackteam 10
B10data = []
directory = 'dataID/BlackTeam/10_Dayon_Griffin'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/BlackTeam/10_Dayon_Griffin/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/BlackTeam/10_Dayon_Griffin/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    B10data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for blackteam 15
B15data = []
directory = 'dataID/BlackTeam/15_Aubrey_Dawkins'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/BlackTeam/15_Aubrey_Dawkins/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/BlackTeam/15_Aubrey_Dawkins/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    B15data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for blackteam 24
B24data = []
directory = 'dataID/BlackTeam/24_Tacko_Fall'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/BlackTeam/24_Tacko_Fall/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/BlackTeam/24_Tacko_Fall/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    B24data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for blackteam 25
B35data = []
directory = 'dataID/BlackTeam/35_Collin_Smith'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/BlackTeam/35_Collin_Smith/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/BlackTeam/35_Collin_Smith/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    B35data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for whiteteam 1
W1data = []
directory = 'dataID/WhiteTeam/1_Zion_Williamson'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/WhiteTeam/1_Zion_Williamson/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/WhiteTeam/1_Zion_Williamson/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    W1data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for whiteteam 3
W3data = []
directory = 'dataID/WhiteTeam/3_Tre_Jones'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/WhiteTeam/3_Tre_Jones/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/WhiteTeam/3_Tre_Jones/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    W3data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for whiteteam 5
W5data = []
directory = 'dataID/WhiteTeam/5_RJ_Barrett'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/WhiteTeam/5_RJ_Barrett/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/WhiteTeam/5_RJ_Barrett/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    W5data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for whiteteam 12
W12data = []
directory = 'dataID/WhiteTeam/12_Javin_DeLaurier'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/WhiteTeam/12_Javin_DeLaurier/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/WhiteTeam/12_Javin_DeLaurier/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    W12data.append(bin30)

# Creating array of 100 sift descriptors + 30-bins for whiteteam 14
W14data = []
directory = 'dataID/WhiteTeam/14_Jordan_Goldwire'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataID/WhiteTeam/14_Jordan_Goldwire/' + images)

        # Convert image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures = 80)
        # Detect keypoints
        kp = sift.detect(gray, None)
        compute, des = sift.compute(gray, kp)

        if len(des) < 0:
            """
            print('dataID/WhiteTeam/14_Jordan_Goldwire/' + images)
            print("WARNING, NOT ENOUGH KEYPOINTS/DESCRIPTORS")
            print("Number of descriptors: ", len(des))
            print("\n")
            """
            pass

        else:
            # Split the image into its respective channels
            chans = cv2.split(image)
            # Initialize the tuple of channel names
            colors = ("b", "g", "r")

            bin30 = [] # 30-bin color histogram
            # Loop over the image channels, the order is BGR
            for (chan, color) in zip(chans, colors):
                # Create a histogram for the current channel
                hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

                for value in hist:
                    bin30 = np.append(bin30, value)
    
    W14data.append(bin30)

# Array containing all data
alldata = np.concatenate((B1data, B10data))
alldata = np.concatenate((alldata, B15data))
alldata = np.concatenate((alldata, B24data))
alldata = np.concatenate((alldata, B35data))

alldata = np.concatenate((alldata, W1data))
alldata = np.concatenate((alldata, W3data))
alldata = np.concatenate((alldata, W5data))
alldata = np.concatenate((alldata, W12data))
alldata = np.concatenate((alldata, W14data))

# Create array of targets
targets = []

# target = B1, means black team player 1
length = len(B1data)
for i in range(length):
    targets.append('B1. BJ Taylor')

length = len(B10data)
for i in range(length):
    targets.append('B10. Dayon Griffin')

length = len(B15data)
for i in range(length):
    targets.append('B15. Aubrey Dawkins')

length = len(B24data)
for i in range(length):
    targets.append('B24. Tacko Fall')

length = len(B35data)
for i in range(length):
    targets.append('B35. Collin Smith')

# target = W1, means white team player 1
length = len(W1data)
for i in range(length):
    targets.append('W1. Zion Williamson')

length = len(W3data)
for i in range(length):
    targets.append('W3. Tre Jones')

length = len(W5data)
for i in range(length):
    targets.append('W5. RJ Barrett')

length = len(W12data)
for i in range(length):
    targets.append('W12. Javin DeLaurier')

length = len(W14data)
for i in range(length):
    targets.append('W14. Jordan Goldwire')

################## DATA CREATION FINISHED - NOW WE TRAIN/TEST THE CLASSIFIER ##################
x_train, x_test, y_train, y_test = train_test_split(alldata, targets, test_size=0.25, random_state=0)
logisticR = LogisticRegression(max_iter=5000)

logisticR.fit(x_train, y_train)
predictions = logisticR.predict(x_test)
score = logisticR.score(x_test, y_test)
print(score)

dump(logisticR, 'models/idclassifier.joblib')

################################# FUNCTION TO TURN IMAGE ARRAY #################################

# Input: image, output: 30-dimensional array representing 30-bin BGR channels (10 bins per channel)
def convertimage(image):
    # Split the image into its respective channels
    chans = cv2.split(image)
    # Initialize the tuple of channel names
    colors = ("b", "g", "r")

    bin30 = [] # 30-bin color histogram
    # Loop over the image channels, the order is BGR
    for (chan, color) in zip(chans, colors):
        # Create a histogram for the current channel
        hist = cv2.calcHist([chan], [0], None, [10], [0, 256])

        for value in hist:
            bin30 = np.append(bin30, value)

    return bin30


"""
##################### MSER ##############
# Create MSER Detector
mser = cv2.MSER_create()

# Detect Regions
regions, _ = mser.detectRegions(image)
print("mser")
print(len(regions))
print(len(regions[0]))
print(len(regions[0][0]))

# Display the image 
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

