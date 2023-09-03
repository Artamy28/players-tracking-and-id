# Identifying and Tracking Basketball Players
**_Images might take some time to load_**  
**_Models and weights were not uploaded because the files were too large_**  
The project can be divided into four sections:  
Player Detection, Court Detection and Tracking, Team Classification and Player Identification, and Player Tracking

## Project Output
![bestoutput](https://github.com/Artamy28/players-tracking-and-id/assets/48444519/f56a4cf9-9848-4e91-b6a5-208815d5ec1a)

## Player Detection
Player detection was achieved by using a YOLOv4 model and training it to detect basketball players.  
Since gathering data is tedious and there were no appropriate public datasets, some automation tools were created to help with this process.  

The training of the model was done using pre-trained weights, which is called transfer learning.  
The final trained model takes a single frame of a video as input and outputs bounding-box coordinates which can be used to draw boxes around detections.  

![player_detection](https://github.com/Artamy28/players-tracking-and-id/assets/48444519/e20f346f-4cb8-417a-a764-2c1361941655)

## Court Detection and Tracking
The court detection system was built so that we could filter detections made outside the basketball court.  
We project a 2D reference court onto the real court using reverse homography.  
This allows us to build a mask that zeroes out all pixel values outside the court so that detections made there are differentiated.

![court_detection_1](https://github.com/Artamy28/players-tracking-and-id/assets/48444519/4e1a11e1-7283-47d8-9c67-37273ed09f6e)
![court_detection_2](https://github.com/Artamy28/players-tracking-and-id/assets/48444519/62c837bf-fd5e-4b26-8e71-562b3c14961a)

## Team Classification and Player Identification
Team classification is performed because it greatly simplifies player identification and player tracking, as the number of targets is reduced by half. Additionally, it also allows us to filter out referees.  

Since players in the same team wear uniforms of the same colour, we use RGB colour histograms to characterize them. So for every image, a 10-bin colour histogram for each of the three RGB channels is computed, resulting in a 30-bin colour histogram.  
So every image is represented by a 30-dimensional feature vector, where each dimension is a positive value. These are then used to train a logistic regression classifier.  

![Screenshot_304](https://github.com/Artamy28/players-tracking-and-id/assets/48444519/02ac86c4-a1a9-46f0-b3c7-6091e6e0d896)
![frame193](https://github.com/Artamy28/players-tracking-and-id/assets/48444519/9363a24a-4385-4133-afbb-825b03bb153d)

## Player Tracking
Player tracking was achieved using DeepSORT, an extension to SORT which is an object-tracking algorithm which uses the method of tracking by detection. This means that objects are detected first and then assigned to tracks.  
On top of this, DEEPSORT introduces a distance metric called the appearance feature vector, which computes the appearance of the object.  
A convolutional neural network classifier is trained, and the final classification layer is removed.  
This leaves us with a dense layer that produces a single feature vector waiting to be classified.  

Our custom feature extractor used in our DeepSORT model was trained on a dataset of basketball players.  
Our final DeepSORT model outputs bounding-box coordinates along with the ID associated with them which can be used to draw boxes around detections and annotate them appropriately.  

![Screenshot_305](https://github.com/Artamy28/players-tracking-and-id/assets/48444519/e9717739-660c-453f-988a-deab8bf586fb)
