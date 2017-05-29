import glob
import matplotlib.pyplot as plt
from functions import *


#IMAGES AND CONSTANTS
vehicle_names_left = glob.glob('../data/model/raw/vehicles/GTI_Left/image*.png')
vehicle_names_right = glob.glob('../data/model/raw/vehicles/GTI_Right/image*.png')
vehicle_names_middle = glob.glob('../data/model/raw/vehicles/GTI_MiddleClose/image*.png')
vehicle_names_far = glob.glob('../data/model/raw/vehicles/GTI_Far/image*.png')
vehicle_names = [vehicle_names_left[0], vehicle_names_right[0], vehicle_names_middle[0], vehicle_names_far[0]]

nonvehicle_names = glob.glob('../data/model/raw/non-vehicles/GTI/image*.png')[:4]
nonvehicle_images = [plt.imread(nonvehicle_name) for nonvehicle_name in nonvehicle_names]
nonvehicle_labels = ["Noncar 1", "Noncar 2", "Noncar 3", "Noncar 4"]

vehicle_labels = ["Left", "Right", "Middle", "Far"]

vehicle_images = [plt.imread(vehicle_name) for vehicle_name in vehicle_names]
gray_images = [make_gray(vehicle_image) for vehicle_image in vehicle_images]
hog_images = [hist_of_gradients(gray_image, orient = 15, pix_per_cell = 8, cell_per_block = 2, vis = True)[1] for gray_image in gray_images]



def plot_car_images():
	fig, axes = plt.subplots(nrows = 2, ncols = 2)
	axes = axes.ravel()
	fig.tight_layout()

	for ax, image, label in zip(axes, vehicle_images, vehicle_labels):
		ax.imshow(image)
		ax.set_title(label)
		ax.axis('off')

	#plt.show()
	plt.savefig('../readme_assets/image.png')

def plot_noncar_images():
	fig, axes = plt.subplots(nrows = 2, ncols = 2)
	axes = axes.ravel()
	fig.tight_layout()

	for ax, image, label in zip(axes, nonvehicle_images, nonvehicle_labels):
		ax.imshow(image)
		ax.set_title(label)
		ax.axis('off')

	#plt.show()
	plt.savefig('../readme_assets/image.png')


def plot_all(images, labels=True):
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (10,3))
	axes = axes.ravel()
	fig.tight_layout()

	for ax, image, label in zip(axes, images, vehicle_labels):
		ax.imshow(image, cmap='gray')
		if labels == True:
			ax.set_title(label)
		ax.axis('off')
	#plt.show()
	plt.savefig('../readme_assets/image.png')
	
#plot_car_images()
#plot_noncar_images()
plot_all(hog_images, labels=False)






