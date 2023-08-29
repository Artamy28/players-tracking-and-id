import cv2
import numpy as np

from joblib import dump, load
import teamclassifier

logisticR = load('models/teamclassifier.joblib')

image = cv2.imread('dataTC/ok/img207.jpg')
data = teamclassifier.convertimage(image)
data = np.reshape(data, (1, 30))

predictions = logisticR.predict(data)
print(predictions)