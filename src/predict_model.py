import pickle
from moviepy.editor import *
from IPython.display import HTML
from functions import *
import warnings
from scipy.ndimage.measurements import label
from collections import deque
warnings.filterwarnings("ignore", category=DeprecationWarning) 

clf = pickle.load(open('../models/classifier.sav', 'rb'))
normalizer = pickle.load(open('../models/normalizer.sav', 'rb'))

def process_frame(frame):
	frame = frame.astype(np.float32)
	true_windows = []

	smallish_windows = slide_window(frame, x_start_stop=[500, None], y_start_stop=[420, 600], xy_window=(48, 48), xy_overlap=(0.5, 0.5))
	for window in smallish_windows:
		window_image = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		X_valid = get_features(window_image)
		X_valid_normalized = normalizer.transform(np.array(X_valid).reshape(1, -1))
		y_valid = clf.predict(X_valid_normalized)
		if y_valid == 1:
			true_windows.append(window)

	small_windows = slide_window(frame, x_start_stop=[600, None], y_start_stop=[420, 600], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
	for window in small_windows:
		window_image = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		X_valid = get_features(window_image)
		X_valid_normalized = normalizer.transform(np.array(X_valid).reshape(1, -1))
		y_valid = clf.predict(X_valid_normalized)
		if y_valid == 1:
			true_windows.append(window)


	medium_windows = slide_window(frame, x_start_stop=[600, None], y_start_stop=[400, 600], xy_window=(96, 96), xy_overlap=(0.5, 0.5))
	for window in medium_windows:
		window_image = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		X_valid = get_features(window_image)
		X_valid_normalized = normalizer.transform(np.array(X_valid).reshape(1, -1))
		y_valid = clf.predict(X_valid_normalized)
		if y_valid == 1:
			true_windows.append(window)


	large_windows = slide_window(frame, x_start_stop=[600, None], y_start_stop=[400, 700], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
	for window in large_windows:
		window_image = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		X_valid = get_features(window_image)
		X_valid_normalized = normalizer.transform(np.array(X_valid).reshape(1, -1))
		y_valid = clf.predict(X_valid_normalized)
		if y_valid == 1:
			true_windows.append(window)


	print("Small windows: ", len(small_windows))
	print("Medium windows: ", len(medium_windows))
	print("Large windows: ", len(large_windows))
	print("True windows: ", len(true_windows))


	heat = np.zeros_like(frame[:,:,0]).astype(np.float)

	for window in true_windows:
		heat = add_heat(heat,true_windows)
	
	heat = apply_threshold(heat, 60)
	heatmap = np.clip(heat, 0, 255)

	labels = label(heatmap)
	draw_trues = draw_boxes(np.copy(frame), true_windows)
	draw_image = draw_labeled_boxes(np.copy(frame), labels)
	draw_image = draw_image * 255.0 #comment for ind frame


	return draw_trues * 255.0



test_images = read_images()
drawn_image = process_frame(test_images[0])
plt.imshow(drawn_image)
plt.show()





