import glob 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import *
from skimage.feature import hog
import numpy as np

#read and store test images
def read_images():
	test_images_names = glob.glob('../data/full_size/test_images/test*.jpg')
	test_images = []
	for test_image_name in test_images_names:
		test_image = mpimg.imread(test_image_name)
		test_images.append(test_image)
	return test_images

#make_gray
def make_gray(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return gray_image

#add heat
def add_heat(heatmap, boxes):
    for box in boxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

#apply thresh
def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

#draw boxes
def draw_boxes(image, boxes, color = (0, 0, 1), thick = 5):
    draw_image = np.copy(image)
    for box_coords in boxes:
        start = box_coords[0]
        end = box_coords[1]
        cv2.rectangle(draw_image, start, end, color, thick)
    return draw_image

#draw labeled boxes
def draw_labeled_boxes(image, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(image, bbox[0], bbox[1], (0,0,1), 6)
    # Return the image
    return image


#make gray
def make_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


#chage colorspace
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

#color histogram feature
def color_hist(image, bins = 16, bins_range = (0,256), vis = False):
	red = np.histogram(image[:,:,0], bins = bins, range = bins_range)
	green = np.histogram(image[:,:,1], bins = bins, range = bins_range)
	blue = np.histogram(image[:,:,2], bins = bins, range = bins_range)
	color_hist_feature = np.concatenate((red[0], green[0], blue[0]))
	return color_hist_feature

#histogram of gradients feature
def hist_of_gradients(img, orient, pix_per_cell, cell_per_block, vis = False):
    if vis == True:
        hog_feature, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                              visualise=True, feature_vector=True)
        return hog_feature, hog_image
    else:      
        hog_feature = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=True)
        return hog_feature

#reduced size feature
def reduce_and_flatten(image, new_size = (32,32)):
	reduced_size_feature = cv2.resize(image, new_size).ravel()
	return reduced_size_feature

#combine features
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

#slide window
def slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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








