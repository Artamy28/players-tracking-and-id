import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
# From images, compute 10-bin color histogram for each of the three RGB channel (resulting in a 30-bin color histogram)

# Creating array of 30-bins for teamA
teamAbins = []
directory = 'dataTC/teamA'
for images in os.listdir(directory):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        image = cv2.imread('dataTC/teamA/' + images)

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
        
        teamAbins.append(bin30)

teamAbins = np.array(teamAbins)

# Creating array of 30-bins for teamB
teamBbins = []
directory = 'dataTC/teamB'
for images in os.listdir(directory):
    # check if the image ends with png
    if (images.endswith(".jpg")):
        image = cv2.imread('dataTC/teamB/' + images)

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
        
        teamBbins.append(bin30)

teamBbins = np.array(teamBbins)

# Creating array of 30-bins for referees
refereebins = []
directory = 'dataTC/referee'
for images in os.listdir(directory):
    # check if the image ends with png
    if (images.endswith(".jpg")):
        image = cv2.imread('dataTC/referee/' + images)

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
        
        refereebins.append(bin30)

refereebins = np.array(refereebins)

# Array containing all bins
allbins = np.concatenate((teamAbins, teamBbins))
allbins = np.concatenate((allbins, refereebins))

# Create array of targets
targets = np.array([])

# target = 0, means teamA
length = teamAbins.shape[0]
for i in range(length):
    targets = np.append(targets, 0)

# target = 1, means teamB
length = teamBbins.shape[0]
for i in range(length):
    targets = np.append(targets, 1)

# target = 2, means referee
length = refereebins.shape[0]
for i in range(length):
    targets = np.append(targets, 2)

################## DATA CREATION FINISHED - NOW WE TRAIN/TEST THE CLASSIFIER ##################
x_train, x_test, y_train, y_test = train_test_split(allbins, targets, test_size=0.25, random_state=0)
logisticR = LogisticRegression(max_iter=1000)

logisticR.fit(x_train, y_train)
predictions = logisticR.predict(x_test)
score = logisticR.score(x_test, y_test)
print(score)

dump(logisticR, 'models/teamclassifier.joblib')

######################### FUNCTION TO TURN IMAGE INTO 30-BIN BGR ARRAY #########################

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