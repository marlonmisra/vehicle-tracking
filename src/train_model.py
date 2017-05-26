import time
import pickle
import numpy as np 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.utils import np_utils

#PARAMS
#both neurals
dropout_prob = 0.8
activation_function = 'softmax'
loss_function = 'categorical_crossentropy'
verbose_level = 1

#simple neural
neural_batches = 64
neural_epochs = 10

#conv neural
convolutional_batches = 64
convolutional_epochs = 10


#READ DATA
#combined features
X_train = np.load('../data/model/processed/X_train.npy')
X_test = np.load('../data/model/processed/X_test.npy')
y_train = np.load('../data/model/processed/y_train.npy')
y_test = np.load('../data/model/processed/y_test.npy')

#convolutional features
X_train_convolutional = np.load('../data/model/processed/X_train_convolutional.npy')
X_test_convolutional = np.load('../data/model/processed/X_test_convolutional.npy')
y_train_convolutional = np.load('../data/model/processed/y_train_convolutional.npy')
y_test_convolutional = np.load('../data/model/processed/y_test_convolutional.npy')


#SVM
def train_SVM():
	clf = LinearSVC()
	t=time.time()
	print("Training samples: ", len(X_train))
	clf.fit(X_train, y_train)
	t2=time.time()
	print("Time (s) to train model:", round(t2-t, 2))

	pred = clf.predict(X_test)
	testing_accuracy = accuracy_score(pred, y_test)
	print("Test Accuracy of SVM: ", testing_accuracy)

	pickle.dump(clf, open("../models/SVM_model.sav", 'wb'))
	print("Classifier saved")


#Simple neural net with combined features
def train_neural():
	y_train_cat = np_utils.to_categorical(y_train, 2) 
	y_test_cat = np_utils.to_categorical(y_test, 2)

	model = Sequential()
	model.add(Dense(128, input_shape=(4884,)))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(64))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(32))
	model.add(Dense(2, activation=activation_function))
	model.summary()
	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
	history = model.fit(X_train, y_train_cat, batch_size=neural_batches, epochs = neural_epochs, verbose = verbose_level, validation_data=(X_test, y_test_cat))
	model.save('../models/neural_model.h5')


#Convolutional neural net with raw image features
def train_convolutional_neural():
	y_train_cat = np_utils.to_categorical(y_train_convolutional, 2) 
	y_test_cat = np_utils.to_categorical(y_test_convolutional, 2)

	model = Sequential()
	model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='valid', input_shape=(64, 64, 3)))
	model.add(MaxPooling2D(pool_size = (3,3)))
	model.add(Flatten())
	#rem
	model.add(Dense(64))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(32))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(16))
	model.add(Dropout(rate=dropout_prob))
	model.add(Dense(8))
	#rem
	model.add(Dense(2, activation=activation_function))
	model.summary()
	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
	history = model.fit(X_train_convolutional, y_train_cat, batch_size=convolutional_batches, epochs = convolutional_epochs, verbose = verbose_level, validation_data=(X_test_convolutional, y_test_cat))
	model.save('../models/convolutional_model.h5')


train_SVM()
#train_neural()
#train_convolutional_neural()








