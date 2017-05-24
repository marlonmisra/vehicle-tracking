## Vehicle tracking project

### Introduction 
Continuing with the self driving car projects, this one is about is about creating a pipeline that looks at dashcam footage and detects and tracks cars. 

The goals/steps are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply color transforms and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained neural network classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)
[image1]: ./output_image/output_image.png "Output image"



### Files and project navigation 
The project includes the following files:
* functions.py contains helper functions
* process_data.py converts the images to feature vectors and saves those in X_test.npy, X_train.npy, y_test.npy, and y_train.npy
* train_model.py contains the main model
* process_frame.py contains the main function that is run on every frame of the video
* pipeline.py reads each frame and applies the process_frame function
* video_annotated.mp4 is the annotated video


### Histogram of gradients
One of the features that I used was a histogram of gradients or HOG. The technique counts occurrences of gradient orientation in localized portions of an image. The image I used as input was a grayscale image - this is sufficient because grayscale images show edges well. The function is defined in functions.py and it is called with parameters in process_data.py. I came up with the following parameters after starting with the provided ones and visually comparing results of various parm combinations. 
* hog_orient = 9
* hog_pix_per_cell = 8
* hog_cell_per_block = 2


### Other features
In addition to a HOG, I also made use of a "mini HLS feature" and a "color histogram". For the former, I utilized the HLS space because exploratory analysis suggested the cars stands out more clearly in this space. I combined that transformation with a reduce_and_flatten function which shrinks the image to 16x16. For the latter, the color histogram was a straightforward RGB histogram with 16 bins. I then concatenated all features (HOG, mini HLS feature, color histogram).

### Processing
After consolidating my data, I shuffled it using the sklearn `shuffle()` function. Then I normalized the data using the sklearn `StandardScaler` function. Finally, I split the data into training/testing sets using the sklearn `train_test_split` function.

### Model
I utilized a straighforward Linear Support Vector Machine (SVM) model from sklearn. After training multiple times, I noticed that modifying the C parameter and making it smaller to 0.005 helped to increase testing accuracy (although training accuracy worsened).

Training was done on 5,380 observations and testing on 1346 observations (20% of total).

### Window search and model tuning
To do the window search, I made use of 4 different window sizes (48 x 48, 64 x 64, 96 x 96, 128 x 128). This was necessary because cars near the bottom of the image will appear much larger than the ones in the back. Because of this fact, I also restricted the search to the y values where it made sense to look. For example, I didn't use the largest 128*128 filter near the top of the image where cars, if they appear, would be much smaller. I also overall restricted the search area to exclude the top part of the image that is mostly consumed by the sky. For every window position, I got the image at at that area and ran the SVM model. For window for which a match was found, I added the window parameters to a list called true_windows. 

Once I had a list of true_windows I made the use of the heat approach to try and find duplicates. Specifically, I created an empty copy of the input image and added 1 to each pixel where a window found a match. Since I used 4 windows the highest value a pixel could get was 4. I then made of a threshold function that set all pixels to 0 where the threshold wasn't found. I found the ideal threshold to be 3 after some testing. Finally, I made use of the labels approach to place boxes around heat zones.

### Video 
The output video is called video_annotated.mp4. An example output image is below. 
![alt text][image1]


### Further discussion 
Thigns I could have improved on:
* Utilized a large dataset that Udacity opensourced.
* Tried to use different models, like neural networks or decision trees.
* Create a class to keep track of frames and do averaging over frames that happen in sequence.










# vehicle-tracking
