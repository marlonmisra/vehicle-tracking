import pickle
from functions import *
from scipy.ndimage.measurements import label
from collections import deque
from build_features import *
from keras.models import load_model

#PARAMS
model_choice = 'svm' #svm, neural, convolutional
heatmap_threshold = -1 #adjust to 3 for image, 20 for video
deque_len = 7
window_overlap = (0.75,0.75)
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


#GLOBAL
heatmaps = deque(maxlen=deque_len)

#LOAD MODELS AND NORMALIZER
clf_svm = pickle.load(open('../models/SVM_model.sav', 'rb'))
clf_neural = load_model('../models/neural_model.h5')
clf_conv = load_model('../models/convolutional_model.h5')
normalizer = pickle.load(open('../models/normalizer.sav', 'rb'))

#PROCEDURE FUNCTIONS
def svm_procedure(img):
    X_valid = get_features(img, use_hog = use_hog_feature, use_color_hist = use_color_hist_feature, use_mini = use_mini_feature, mini_color_space = mini_clr_space , mini_size = mini_dimension, hog_orient = hog_orientations, hog_pix_per_cell = hog_pixels_per_cell, hog_cell_per_block = hog_cells_for_each_block)
    X_valid_normalized = normalizer.transform(X_valid.reshape(1,-1))
    y_valid = clf_svm.predict(X_valid_normalized)
    if y_valid == 1:
        return True

def neural_procedure(img):
    X_valid = get_features(img, use_hog = use_hog_feature, use_color_hist = use_color_hist_feature, use_mini = use_mini_feature, mini_color_space = mini_clr_space , mini_size = mini_dimension, hog_orient = hog_orientations, hog_pix_per_cell = hog_pixels_per_cell, hog_cell_per_block = hog_cells_for_each_block)
    X_valid_normalized = normalizer.transform(X_valid.reshape(1,-1))
    y_valid = clf_neural.predict(X_valid_normalized)
    if y_valid[0][1] > 0.5:
        return True

def convolutional_procedure(img):
    X_valid = np.array(img)
    X_valid = X_valid[None, :]
    y_valid = clf_conv.predict(X_valid)
    if y_valid[0][1] > 0.5:
        return True


#PROCESS FRAME
def process_frame(frame, model_type = 'svm', heatmap_thresh = heatmap_threshold, all_outputs=False):
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
		heatmap_2 = apply_threshold(np.copy(heatmap), heatmap_thresh)
		labels = label(heatmap_2) #tuple with 1st element color-coded heatmap and second elment int with number of cars
		image_final = draw_labeled_boxes(np.copy(frame), labels)
	else:
		image_final = np.copy(frame)
	print('length', len(heatmaps))
	if all_outputs == True:
		return image_windows, heatmap, heatmap_2, labels[0], image_final
	else:
		return image_final #* 255 for video

#test_images = read_images()
#a = process_frame(test_images[0], model_type = model_choice, heatmap_thresh = -1)
#plt.imshow(a)
#plt.show()






