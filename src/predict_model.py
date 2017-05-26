import pickle
from functions import *
from scipy.ndimage.measurements import label
from collections import deque
from build_features import *
from keras.models import load_model

#PARAMS
model_choice = 'svm' #svm, neural, convolutional
heatmap_threshold = 10
smallest_window_size = (48, 48)
small_window_size = (64, 64)
medium_window_size = (96, 96)
large_window_size = (128, 128)
window_overlap = (0.5,0.5)


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
def process_frame(frame, model_type = 'svm'):
	frame = frame.astype(np.float32)
	frame /= 255.0
	
	smallest_windows = slide_window(frame, x_start_stop=[500, None], y_start_stop=[300, 500], xy_window=smallest_window_size, xy_overlap=window_overlap)
	small_windows = slide_window(frame, x_start_stop=[0, None], y_start_stop=[350, 550], xy_window=small_window_size, xy_overlap=window_overlap)
	medium_windows = slide_window(frame, x_start_stop=[0, None], y_start_stop=[400, 600], xy_window=medium_window_size, xy_overlap=window_overlap)
	large_windows = slide_window(frame, x_start_stop=[0, None], y_start_stop=[450, 700], xy_window=large_window_size, xy_overlap=window_overlap)

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

	print("True windows: ", len(true_windows))
	image_windows = draw_boxes(np.copy(frame), true_windows)
	heat = np.zeros_like(frame[:,:,0]).astype(np.float)
	for window in true_windows:
		heat = add_heat(heat,true_windows)
	heatmap_1= apply_threshold(heat, heatmap_threshold)
	heatmap_2 = np.clip(heatmap_1, 0, 1)
	labels = label(heatmap_2) #tuple with 1st element color-coded heatmap and second elment int with number of cars
	image_final = draw_labeled_boxes(np.copy(frame), labels)
	image_final = image_final * 255 #for video

	return image_final



#test_images = read_images()
#drawn_image = process_frame(test_images[0], model_type = model_choice)
#plt.imshow(drawn_image)
#plt.show()





