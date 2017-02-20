# Import the modules
from __future__ import division
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Load the dataset
images = np.load('data/poolImagesTest.npy')
labels = np.load('data/poolLabelsTest.npy')
images = np.concatenate((images,np.load('data/notPoolImagesTest.npy')),axis=0)
labels = np.hstack((labels,np.load('data/notPoolLabelsTest.npy')))

# Extract the features and labels
features = np.array(images, 'int16') 
labels = np.array(labels, 'int')

ii = 0
correctPrediction = 0
for feature in features:
	fd0 = hog(feature[:,:,0].reshape((24, 24)), orientations=9, pixels_per_cell=(12, 12), cells_per_block=(1, 1), visualise=False)
	fd1 = hog(feature[:,:,1].reshape((24, 24)), orientations=9, pixels_per_cell=(12, 12), cells_per_block=(1, 1), visualise=False)
	fd2 = hog(feature[:,:,2].reshape((24, 24)), orientations=9, pixels_per_cell=(12, 12), cells_per_block=(1, 1), visualise=False)
	fd = np.concatenate((fd0,fd1,fd2))
	nbr = clf.predict(np.array([fd], 'float64'))
	if nbr == labels[ii]:
		correctPrediction+=1
	ii+=1
print correctPrediction/ii