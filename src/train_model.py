import numpy as np 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

def train_model():
	print("Loading data...")
	X_train = np.load('../data/model/processed/X_train.npy')
	y_train = np.load('../data/model/processed/y_train.npy')
	print("Finished loading training data")
	X_test = np.load('../data/model/processed/X_test.npy')
	y_test = np.load('../data/model/processed/y_test.npy')
	print("Finished loading testing data")

	clf = LinearSVC()
	t=time.time()
	print("Training samples: ", len(X_train))
	clf.fit(X_train, y_train)
	t2=time.time()
	print("Time (s) to train model:", round(t2-t, 2))

	pred = clf.predict(X_test)
	testing_accuracy = accuracy_score(pred, y_test)
	print("Test Accuracy of SVM: ", testing_accuracy)

	pickle.dump(clf, open("../models/classifier.sav", 'wb'))
	print("Classifier saved")


train_model()







"""

X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')

y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)
print(X_train.shape)

model = Sequential()
model.add(Dense(128, input_dim=(4884)))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, nb_epoch = 1, verbose = 1, validation_data=(X_test, y_test))

model.save('../models/my_model.h5')

"""
