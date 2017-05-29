## Vehicle tracking project

### Introduction 
After building two lane detection pipelines ([simple](https://github.com/marlonmisra/lane-finding), [advanced](https://github.com/marlonmisra/advanced-lane-finding)), this project is about another fundamental problem in self driving cars - the detection other cars. Again, we're using a single front facing car camera for our input video feed. The output is an annotated version of the input feed that includes rectangles around identified cars. For this project, we're strictly focused on the detection of other cars, but the model can easily be trained on trucks, humans, traffic signs, or other objects. 

The steps I'll describe are: 
* Exploring different feature engineering techniques, including using a histogram of oriented gradients (HOG), a histogram of color, and reduced flattened version of the original image. 
* Comparing different classifiers, including a Support Vector Machine (SVM), a simple neural net, and a convolutional neural net.
* Implementing a sliding-window technique where areas of the input image are iteratively searched for the presence of a car car.
* Running the pipeline on a video stream and making use of prior frames to reduce false positives. 


[//]: # (Image References)

[image1]: ./readme_assets/car_images.png "car images"
[image2]: ./readme_assets/notcar_images.png "noncar images"
[image3]: ./readme_assets/full_test_image.jpg "full image"
[image4]: ./readme_assets/vehicle_images.png "vehicle images"
[image5]: ./readme_assets/gray_imgs.png "gray images"
[image6]: ./readme_assets/hog_imgs.png "hog images"

[image7]: ./readme_assets/full_images.png "full images"
[image8]: ./readme_assets/positive_detections.png "positive detections"
[image9]: ./readme_assets/positive_detections_labels.png "positive detections label"
[image10]: ./readme_assets/heatmaps.png "heatmaps"
[image11]: ./readme_assets/heatmaps_labels.png "heatmaps labels"
[image12]: ./readme_assets/thresholded_heatmaps.png "thresholded heatmaps"
[image13]: ./readme_assets/thresholded_heatmaps_labels.png "thresholded heatmaps labels"
[image14]: ./readme_assets/labels.png "labels"
[image15]: ./readme_assets/labels_labels.png "labels labels"
[image16]: ./readme_assets/final_images.png "final images"
[image17]: ./readme_assets/final_images_labels.png "final images labels"





### Files and project navigation 
The project includes the following files:
* The data/model folder contains image data for training (vehicle or not a vehicle) - it contains both the raw image data and the processed image data that was saved after feature extraction. The data/full_size folder contains test images and test videos that are full size (i.e. the front-facing car camera data).
* The models folder contains saved parameters for the SVM classifier, the simple neural net classifier, the convolutional neural net classifier, and a normalizer. 
* The results folder contains annotated results for test images and test videos. 
* The src folder contains all Python scripts. Specifically build_features.py for feature extraction, functions.py for helper functions, pipeline.py for video processing, predict\_model.py for model predictions, and train\_model.py for model training. 


### Data 

The raw data consists of 64x64 images which are labeled as either car or not car. Car images further break down into images from the left, right, middle, and far. In total there 10,885 samples, of which 1,917 (17.6%) are labeled as car, and the remainder (82.4%) are labeled as not car. We're going to start by extracting features from this data. Below are examples of car images and noncar images. 

![alt text][image1]

![alt text][image2]


Note that the above data what will be used for model training and testing. Our ultimate goal, however, is to detect cars on full images like the one below. 

![alt text][image3]

### Feature extraction

**Background**

For traditional classifiers like Support Vector Machines, it's important to first do feature engineering. If you don't, the features are too large and the model will take too long to train. There has also been lots of work done to figure out what good derived features are, so we are going to implement some of those. 

**Histogram of oriented gradients (HOG)**

HOG's are known to be good predictors of object presence. They work by first calculating the gradient magnitude and direction at each pixel. Then, the individual pixel values are grouped into cells of size x*x. Then, for each cell a histogram of gradient directions is computed (where magnitude is also considered). When you do this for all cells and plot a result you begin to see a representation of the original structure, and that is what a HOG is. The reason this feature is so useful is because it's robust to changes in color and small variations in shape. To implement, I used the `hog()` function from skimage.feature. The parameters I used are below.  

I've also plotted the original images, grey images (you convert to grayscale before applying the hog function), and the hog images. Note that the hog images are only visualizations - the feature is a flattened out vector of that representation.

```python
hog_orientations = 15
hog_pixels_per_cell = 8
```

![alt text][image4]
![alt text][image5]
![alt text][image6]

**Color histogram**

The color histogram is the second feature we're going to make use of. It complements the HOG well because rather than focusing on edge distribution, it's focused on color distribution. The function to extract the color features was defined as follows. I used 16 bins of colors. 

```python
def color_hist(image, bins = 16, bins_range = (0,256), vis = False):
	red = np.histogram(image[:,:,0], bins = bins, range = bins_range)
	green = np.histogram(image[:,:,1], bins = bins, range = bins_range)
	blue = np.histogram(image[:,:,2], bins = bins, range = bins_range)
	color_hist_feature = np.concatenate((red[0], green[0], blue[0]))
	return color_hist_feature
```

**Colorspace-transformed reduced image**

The purpose of this feature is to capture as much information as possible from the original raw image, while significantly reducing feature size. The approach I used to derive this feature is to (1) convert the RGB image to a colorspace that highlighted edges, (2) resize the image to be smaller and (3) flatten the image. The two functions below were applied in sequence with the following parameters. 

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

The previous 3 feature extraction techniques each returned a vector. To combine them, I simply appended them to one another. I defined the combining function in a way that lets me exclude any one of the three features, so that I can test if that improves accuracy. 

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

**Normalizing, randomizing, and splitting**

We now have combined feature vector which we created by combining 3 separate features. Rather than using these directly, since they each have different relative magnitudes, I normalized them using the sklearn.preprocessing `StandardScaler` module. After that, I used the sklearn.model_select `train_test_split()` function to randomize and split the data into training and testing sets. I dedicated 20% of all observations to testing.

```python
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X) 
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=test_fraction, random_state=42)
```


### Classifiers and training

**Approach 1 - Support Vector Machine with derived features**

I implemented the SVM using the Keras LinearSVM module and utilized GridSearchCV to test a range of C parameters. The C parameter tells the SVM optimization how important it is to avoid misclassifying each training example. For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of classifying all the training points. Conversely, a small C value will cause the optimizer to look for a larger-margin separating hyperplane, even if that implies more misclassification. 

I was able to achieve an accuracy of 98.62% using C = 0.01. 

```python
def train_SVM():
	params = {'C':[0.01, 0.1, 1]}
	clf = GridSearchCV(LinearSVC(), params)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	testing_accuracy = accuracy_score(pred, y_test)
	pickle.dump(clf, open("../models/SVM_model.sav", 'wb'))
	print("Classifier saved")
```


**Approach 2 - Neural network with derived features**

The second technique I tried was a simple fully connected neural network. I still used the derived features because non-convolutional networks are not good at doing feature extraction on images on their own. 

I experimented with different network architectures, layer types, and parameters. Ultimately, I found dense layers to work best, coupled with Dropout regularization layers to teach the model redundancy. To introduce nonlinearity into the network I used standard relu function except for the last layer where I used a softmax activation function so that I can use categorical crossentropy as the loss function. 

With the setup below I was able to achieve a testing accuracy of 99.45%. 

```python
dropout_prob = 0.6
activation_function = 'relu'
loss_function = 'categorical_crossentropy'
neural_batches = 64
neural_epochs = 10 
```

```python
def train_neural():
	y_train_cat = np_utils.to_categorical(y_train, 2) 
	y_test_cat = np_utils.to_categorical(y_test, 2)
	model = Sequential()
	model.add(Dense(32, input_shape=(6060,)))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(32, activation=activation_function))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(32, activation=activation_function))
	model.add(Dense(2, activation='softmax'))
	model.summary()
	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
	history = model.fit(X_train, y_train_cat, batch_size=neural_batches, epochs = neural_epochs, verbose = verbose_level, validation_data=(X_test, y_test_cat))
	model.save('../models/neural_model.h5')
```


**Approach 3 - Convolutional neural network with raw features**

For the last approach I used raw features instead of the derived ones. I did this because convolutional layers do feature extraction on images really well and I wanted to see how well raw features worked compared to the derived ones. After the convolutional layers, I used a MaxPooling2D layer to squeeze the spatial dimensions and reduce complexity, so that training runs faster. 

With the setup below I was able to achieve a testing accuracy of 99.08%. The convolutional neural net took by far the longest to train at at 960 seconds. It also took more epochs (25) for accuracies to get into the high 90s range, whereas that happened much faster for standard covnets. 

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
	model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='valid'))
	model.add(MaxPooling2D(pool_size = (3,3)))
	model.add(Dropout(rate=dropout_prob))
	model.add(Flatten())
	model.add(Dense(64,activation=activation_function))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(32,activation=activation_function))
	model.add(Dense(32,activation=activation_function))
	model.add(Dense(2,activation='softmax'))
	model.summary()
	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
	history = model.fit(X_train_convolutional, y_train_cat, batch_size=convolutional_batches, epochs = convolutional_epochs, verbose = verbose_level, validation_data=(X_test_convolutional, y_test_cat))
	model.save('../models/convolutional_model.h5')
```


### Window search and model tuning

**Introduction**

Now that the classifiers are saved, the next step is to come up with a technique to apply them. Recall that model training was done on 64*64 images which were labeled car or not car. However, the images that the pipeline runs on are coming off of the front-facing camera. This means the image is much larger and contains many more things (the sky, parallel lanes, other cars, etc.). 

To deal with that, we're going to use a window search technique where we iteratively shift a window over sections of the image and do a search for a car at that location. The function that performs the search is below. 


```python
def slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], y_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = image.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = image.shape[0]
    #span of regions 
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    #number of pixels
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    #number of windows
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            #window positions
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list
```

**Window parameters**

This above function works on window sizes of arbitrary dimensions. This is important because when the search happens near the bottom of the image (where other cars are close), the windows have to be larger because cars closer to your car appear to be larger. Conversely, cars that are higher in the image (further away in real life) require smaller windows. Because of that, we defined 4 different window sizes. 

There are other things we can specify as well. First, we can restrict the area to search in. For example, we don't care if any cars are detected at the very top of the image or the far sides. We can also specify how much the windows shift in each iteration by setting the `window_overlap` parameter. Images of all windows that found matches are below.

```python
smallest_window_size = (48, 48)
small_window_size = (64, 64)
medium_window_size = (96, 96)
large_window_size = (128, 128)

smallest_x_start_stop = [500, None]
small_x_start_stop = [0, None]
medium_x_start_stop = [0, None]
large_x_start_stop = [0, None]
smallest_y_start_stop = [320, 500]
small_y_start_stop = [350, 550]
medium_y_start_stop = [400, 600]
large_y_start_stop = [450, 700]

window_overlap = (0.75,0.75)
```


![alt text][image7]

![alt text][image8]


**Heatmaps**

Now that we have "true windows" or areas of the image where our classifier detected cars, we want to remove false positives. To do that, we're going to make use of heatmaps.

Starting with a copy of the original image where each pixel is set to 0, we use the `add_heat()` function and add 1 to any any pixel that is covered by a true window. Since there are multiple size windows and overlaps can occur, on the heatmap (2nd row of images), some areas are hotter than others. Next, we apply a threshold to this heatmap so that areas where few windows overlap are exluded. Mathematically, we set those values back to 0. That's why on thresholded heatmap (3rd row of images), there are fewer hot areas. Then, as the last step, we make use of the scipy.ndimage.measurements `label()` function to turn our thresholded heatmap into discrete areas. The output of that is represented in the 4th row of images. 

In the actual video pipeline, I've increased the heatmap_threshold and also made of a heatmaps deque. The purpose of this is to keep track of multiple frames and do smoothing. This works really well because false positive usually don't occur consistently over multiple frames. 

```python
heatmap_threshold = 2
heatmaps = deque(maxlen=deque_len)
deque_len = 7
```

![alt text][image9]

![alt text][image10]

![alt text][image12]

![alt text][image14]

![alt text][image16]


```python
def process_frame(frame, model_type = 'svm'):
	frame = frame.astype(np.float32)
	frame /= 255.0
	
	smallest_windows = slide_window(frame, x_start_stop=smallest_x_start_stop, y_start_stop=smallest_y_start_stop, xy_window=smallest_window_size, xy_overlap=window_overlap)
	small_windows = slide_window(frame, x_start_stop=small_x_start_stop, y_start_stop=small_y_start_stop, xy_window=small_window_size, xy_overlap=window_overlap)
	medium_windows = slide_window(frame, x_start_stop=medium_x_start_stop, y_start_stop=medium_y_start_stop, xy_window=medium_window_size, xy_overlap=window_overlap)
	large_windows = slide_window(frame, x_start_stop=large_x_start_stop, y_start_stop=large_y_start_stop, xy_window=large_window_size, xy_overlap=window_overlap)

	window_types = [smallest_windows, small_windows, medium_windows, large_windows]
	true_windows = []
	for window_set in window_types:
		for window in window_set:
			window_image = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
			if model_type == 'svm':
				if svm_procedure(window_image):
					true_windows.append(window)
			if model_type == 'neural':
				if neural_procedure(window_image):
					true_windows.append(window)
			if model_type == 'convolutional':
				if convolutional_procedure(window_image):
					true_windows.append(window)
	
	image_windows = draw_boxes(np.copy(frame), true_windows)
	print("True windows: ", len(true_windows))

	heatmap = np.zeros_like(frame[:,:,0]).astype(np.float)
	if len(true_windows)>0:
		heatmap = add_heat(heatmap, true_windows)
		
		heatmaps.append(heatmap)
		if len(heatmaps)==deque_len:
			heatmap = sum(heatmaps)
		heatmap_2 = apply_threshold(np.copy(heatmap), heatmap_threshold)
		labels = label(heatmap_2) #tuple with 1st element color-coded heatmap and second elment int with number of cars
		image_final = draw_labeled_boxes(np.copy(frame), labels)
	else:
		image_final = np.copy(frame)
	print('length', len(heatmaps))
	return image_final * 255
```

### Video pipeline
The `pipeline.py` file is located in the src folder. The `process_video(input_path, output_path)` defined here calls the `process_frame()` function in the predict_model.py file and executes the function on every frame of the video. 


### Discussion
The pipeline generally works well After trying various techniques to remove false positives, the multi-frame heatmap technique with the right parameters seemed to be the most helpful. 

In the future, improvements I'd like to make are: 
* Generalizing the pipeline to also detect humans and traffic lights.
* Generating new data by rotating existing data and doing transforms.

