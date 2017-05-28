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


### Feature extraction

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

**Colorspace-transformed reduced image**

The purpose of this feature is to capture as much as possible from the raw image, while still significantly reducing complexity. The approach to derive this feature is to (1) convert the RGB image to a more useful colorspace, (2) resize the image to be smaller and (3) flatten the image. The two functions below were applied in sequence with the following parameters. 

```python
color_space = 'YCrCb'
new_size = (32, 32)
```

```python
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
```

**Combining the features into a single feature**

The previous 3 feature extraction techniques each returned a vector. This means that to combine them, we can simply append them to each other. The function I defined to this is below. I defined it in a way that lets me exclude any one of the three features, so that I can test if that improves accuracy. 

```python
def get_features(image, use_hog = True, use_color_hist = True, use_mini = True, mini_color_space = 'HSV', mini_size = (32,32), hog_orient = 9, hog_pix_per_cell = 8, hog_cell_per_block = 2):
    X = np.array([])
    if use_hog == True:
        hog_feature = hist_of_gradients(make_gray(image), orient = hog_orient, pix_per_cell = hog_pix_per_cell, cell_per_block = hog_cell_per_block, vis = False)
        X = np.append(X, hog_feature)
    if use_color_hist == True:
        color_histogram_feature = color_hist(image)
        X = np.append(X, color_histogram_feature)
    if use_mini == True:
        mini_feature = reduce_and_flatten(change_colorspace(image, color_space = mini_color_space), new_size = mini_size)
        X = np.append(X, mini_feature)
    return X
```

### Classifiers and training

**Approach 1 - Support Vector Machine with derived features**

I implemented the SVM using the Keras LinearSVM module. I also utilized GridSearchCV to test a range of C parameters. The C parameter tells the SVM optimization how important it is to avoid misclassifying each training example. For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of classifying all the training points. Conversely, a small C value will cause the optimizer to look for a larger-margin separating hyperplane, even if that implies more misclassification. 

I was able to achieve an accuracy of 98.1% using C = 0.01. 

```python
def train_SVM():
	params = {'C':[0.001, 0.01, 0.1, 1, 10]}
	clf = GridSearchCV(LinearSVC(), params)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	testing_accuracy = accuracy_score(pred, y_test)
	pickle.dump(clf, open("../models/SVM_model.sav", 'wb'))
	print("Classifier saved")
```


**Approach 2 - Neural network with derived features**
The second technique I tried was a standard neural network. I still used the derived features because I figured the raw features were best left to the the last approach (a conv. neural network).

I experimented with different network architectures, layer types, and parameters. Ultimately, I found standard dense layers to work best, coupled with Dropout regularization layers to teach the model redundancy. I used a softmax activation function on the last layer so that I could use categorical crossentropy as the loss function. 

With the setup below I was able to achieve a testing accuracy of 97.6%. 

```python
dropout_prob = 0.6
loss_function = 'categorical_crossentropy'
neural_batches = 64
neural_epochs = 10 
```

```python
def train_neural():
	y_train_cat = np_utils.to_categorical(y_train, 2) 
	y_test_cat = np_utils.to_categorical(y_test, 2)
	model = Sequential()
	model.add(Dense(128, input_shape=(4884,)))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(64))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(32))
	model.add(Dense(2, activation=softmax))
	model.summary()
	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
	history = model.fit(X_train, y_train_cat, batch_size=neural_batches, epochs = neural_epochs, verbose = verbose_level, validation_data=(X_test, y_test_cat))
	model.save('../models/neural_model.h5')
```



**Approach 3 - Convolutional neural network with raw features**
For this last approach I decided to use raw features rather than the derived features. I did this because the whole point of a convolutional layer is to be able to handle and make sense of 3 dimensional images inputs and derive better features for successive layers. Immidiately after the convolutional layer, I also made use of a MaxPooling2D layer to squeeze the spatial dimensions and reduce commplexity. 

With the setup below I was able to achieve a testing accuracy of 97.6%.

```python
dropout_prob = 0.6
activation_function = 'relu'
loss_function = 'categorical_crossentropy'
convolutional_batches = 64
convolutional_epochs = 25
```

```python
def train_convolutional_neural():
	y_train_cat = np_utils.to_categorical(y_train_convolutional, 2) 
	y_test_cat = np_utils.to_categorical(y_test_convolutional, 2)
	model = Sequential()
	model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='valid', input_shape=(64, 64, 3)))
	model.add(MaxPooling2D(pool_size = (3,3)))
	model.add(Flatten())
	model.add(Dense(128,activation=activation_function))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(64,activation=activation_function))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(32,activation=activation_function))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(16,activation=activation_function))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(2,activation='softmax'))
	model.summary()
	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
	history = model.fit(X_train_convolutional, y_train_cat, batch_size=convolutional_batches, epochs = convolutional_epochs, verbose = verbose_level, validation_data=(X_test_convolutional, y_test_cat))
	model.save('../models/convolutional_model.h5')
```






-----------

### Histogram of gradients
One of the features that I used was a histogram of gradients or HOG. The technique counts occurrences of gradient orientation in localized portions of an image. The image I used as input was a grayscale image - this is sufficient because grayscale images show edges well. The function is defined in functions.py and it is called with parameters in process_data.py. I came up with the following parameters after starting with the provided ones and visually comparing results of various parm combinations. 
* hog_orient = 9
* hog_pix_per_cell = 8
* hog_cell_per_block = 2

### Processing
After consolidating my data, I shuffled it using the sklearn `shuffle()` function. Then I normalized the data using the sklearn `StandardScaler` function. Finally, I split the data into training/testing sets using the sklearn `train_test_split` function.


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
