from functions import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def process():	
	#create lists for images and labels
	images = []
	y = []

	print("Starting to process data...")
	#get non-vehicle images and labels
	non_vehicle_names = glob.glob('../data/model/raw/non-vehicles/GTI/image*.png')
	for non_vehicle_name in non_vehicle_names:
		non_vehicle_image = plt.imread(non_vehicle_name)
		images.append(non_vehicle_image)
		y.append(0)
	print("Finished reading non-car data")

	#get vehicle images and labels
	vehicle_names_far = glob.glob('../data/model/raw/vehicles/GTI_Far/image*.png')
	for vehicle_name in vehicle_names_far:
		vehicle_image = plt.imread(vehicle_name)
		images.append(vehicle_image)
		y.append(1)

	vehicle_names_left = glob.glob('../data/model/rawvehicles/GTI_Left/image*.png')
	for vehicle_name in vehicle_names_left:
		vehicle_image = plt.imread(vehicle_name)
		images.append(vehicle_image)
		y.append(1)

	vehicle_names_middle = glob.glob('../data/model/raw/vehicles/GTI_MiddleClose/image*.png')
	for vehicle_name in vehicle_names_middle:
		vehicle_image = plt.imread(vehicle_name)
		images.append(vehicle_image)
		y.append(1)

	vehicle_names_right = glob.glob('../data/model/raw/vehicles/GTI_Right/image*.png')
	for vehicle_name in vehicle_names_right:
		vehicle_image = plt.imread(vehicle_name)
		images.append(vehicle_image)
		y.append(1)
	print("Finished reading car data")

	#turn images into features
	X = []
	for image in images:
		feature = get_features(image, mini_color_space = 'HLS', mini_size = (32,32), hog_orient = 9, hog_pix_per_cell = 8, hog_cell_per_block = 2)
		X.append(feature)
	print("Finished converting image data into features")

	#convert to arrays
	X = np.array(X)
	y = np.array(y)

	print("Total observations: ", len(X))

	#shuffle
	X, y = shuffle(X, y)
	print("Finished shuffling data")

	#normalize
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)
	print("Finished normalizing data")

	#save normalizer for pipeline
	pickle.dump(X_scaler, open("../models/normalizer.sav", 'wb'))
	print("Normalizer saved")

	#train and testing set
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
	print("Finished splitting data")

	print("Training observations: ", len(X_train))
	print("Testing observations: ", len(X_test))

	#save to files
	np.save('../data/model/processed/X_train.npy', X_train)
	np.save('../data/model/processed/y_train.npy', y_train)
	np.save('../data/model/processed/X_test.npy', X_test)
	np.save('../data/model/processed/y_test.npy', y_test)
	print("Finished saving data")

process()













