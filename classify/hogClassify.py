# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter

# Load the dataset
#dataset = datasets.fetch_mldata("MNIST Original")
images = np.load('data/poolImagesTrain.npy')
labels = np.load('data/poolLabelsTrain.npy')
images = np.concatenate((images,np.load('data/notPoolImagesTrain.npy')),axis=0)
labels = np.hstack((labels,np.load('data/notPoolLabelsTrain.npy')))


# Extract the features and labels
features = np.array(images, 'int16') 
labels = np.array(labels, 'int')



# Extract the hog features
list_hog_fd = []
for feature in features:
	fd0 = hog(feature[:,:,0].reshape((24, 24)), orientations=9, pixels_per_cell=(12, 12), cells_per_block=(1, 1), visualise=False)
	fd1 = hog(feature[:,:,1].reshape((24, 24)), orientations=9, pixels_per_cell=(12, 12), cells_per_block=(1, 1), visualise=False)
	fd2 = hog(feature[:,:,2].reshape((24, 24)), orientations=9, pixels_per_cell=(12, 12), cells_per_block=(1, 1), visualise=False)
	fd = np.concatenate((fd0,fd1,fd2))
	list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')


print "Count of digits in dataset", Counter(labels)

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)