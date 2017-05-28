## Vehicle tracking project

### Introduction 
After building two lane detection pipelines (one simple, one advanced), this project is about another fundamental problem in self driving cars - the detection other objects. Again, we're using a single front facing car camera for our input video feed. The output is an annotated version of the input feed that includes rectangles around the objects. For this project, we're strictly focused on the detection of other cars, but the model can easily be trained on trucks, humans, traffic signs, or other objects. 

The steps I'll describe are: 
* Exploring different feature engineering techniques, including using a histogram of oriented gradients (HOG), a histogram of color, and minimization and flattening of the full input image. 
* Comparing different classifiers, including a Support Vector Machine (SVM), a simple neural net, and a convolutional neural (fow which I used raw image inputs)
* Implementing a sliding-window technique to move across the input image and search for cars using the trained classifier. 
* Running the pipeline on a video stream and making use of prior frames to reduce false positives. 
* Estimating boundinging boxes for detected vehicles


[//]: # (Image References)


### Files and project navigation 
The project includes the following files:
* The data/model folder contains image data for training (vehicle or not a vehicle) - it contains both the raw image data and the processed image data that was saved after feature extraction. The data/full_size folder contains test images and test videos that are full size (i.e. the front-facing car camera data).
* The models folder contains saved parameters for the SVM classifier, the simple neural net classifier, the convolutional neural net classifier, and a normalizer. 
* The results folder contains annotated results for test images and test videos. 
* The src folder contains all Python scripts. Specifically build_features.py for feature extraction, functions.py for helper functions, pipeline.py for video processing, predict\_model.py for model predictions, and train\_model.py for model training. 


### Data 

Our raw data consists of 64x64 images which are labeled as either car or not car. In total we have 10,885 samples, of which 1,917 (17.6%) are labeled as car, and the remainder (82.4%) are labeled as not car. We're going to prepare two separate sets of features with this data - the first for a convolutional neural network, and the second for other classifiers.

### Data processing for convolutional neural networks

Preparing data for a convolutional network is straighforward because you're not required to do much feature engineering - instead, the first few layers of the neural net will do that for you (edge detection, etc.). As a result, the function `build_conv_features(images_list, labels_list` is really simple - it just takes each image and converrts it to a NumPy array. Then, it splits the data into training and testing sets.

### Data processing for other classifiers

**Background**
We also want to try other models like Support Vector Machines. For classifiers like that it's important to first do feature engineering. Otherwise the data is too complex and the model will take too long to train. In addition, many feature extraction techniques have been explored by others and are proven to work well for this type of task. 

**Histogram of oriented gradients (HOG)**
HOG's are known to be good predictors of object presence. They work by first calculating the gradient magnitude and direction at each pixel. Then, the individual pixel values are grouped into cells of size x*x. Then, for each cell a histogram of gradient directions is computed (where magnitude is also considered). When you do this for all cells and plot a result you begin to see a representation of the original structure, and that is what a HOG is. The reason this feature is so useful is because it's robust to changes in color and small variations in shape. To implement, I used the `hog()` function from skimage.feature. The output is a vector which I later combine to the other features. The following parameters worked best for me.  

```python
hog_orientations = 15
hog_pixels_per_cell = 8
```

**Color histogram**
The color histogram is similar to the HOG. I decided to include it so that the model could make use of color information. The function to extract the color features was defined as follows.

```python
def color_hist(image, bins = 16, bins_range = (0,256), vis = False):
	red = np.histogram(image[:,:,0], bins = bins, range = bins_range)
	green = np.histogram(image[:,:,1], bins = bins, range = bins_range)
	blue = np.histogram(image[:,:,2], bins = bins, range = bins_range)
	color_hist_feature = np.concatenate((red[0], green[0], blue[0]))
	return color_hist_feature
```

**Colorspace-transformed flattened reduced image**
The purpose of this feature is to capture as much as possible from the raw image, while still significantly reducing complexity. The approach to derive this feature is to (1) convert the RGB image to a more useful colorspace, (2) resize the image to be smaller and (3) flatten the image. The two functions below were applied in sequence with the following parameters. 

```python
color_space = 'YCrCb'
new_size = (32, 32)
```

def change_colorspace(image, color_space):
    if color_space == 'HSV':
        new_colorspace = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'HLS':
        new_colorspace = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)        
    elif color_space == 'LUV':
        new_colorspace = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'YUV':
        new_colorspace = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        new_colorspace = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
    	print("Wrong colorspace selected")

    return new_colorspace 

def reduce_and_flatten(image, new_size = (32,32)):
	reduced_size_feature = cv2.resize(image, new_size).ravel()
	return reduced_size_feature


**Combining the features into a single feature**
The previous 3 feature extraction techniques.






Feature extraction takes the raw image data we have and builds derives features that have high predictive power and less complexity. 

**Approach**

**Approach**

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
