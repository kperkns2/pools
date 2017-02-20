import numpy as np
from scipy import misc
import os 
from matplotlib import pyplot as plt
import imutils
import random
images = np.zeros((15000,24,24,3))
labels = []
iii = 0
for root, dir, files in os.walk("poolTrain"):
	for file in files:
		if file[-4:-1]==".jp":
			im = misc.imread("poolTrain/"+file)
			try:
				images[iii,:,:,:] = imutils.resize(im, width=24)
				labels.append(1)
				iii+=1
				if iii%100==0:
					print iii
			except Exception, e:
				print e

images2 = images[0:9000]
labels2 = labels[0:9000]
		
images = np.zeros((15000,24,24,3))
labels = []
iii = 0
for root, dir, files in os.walk("notTrain"):
	for file in files:
		if file[-4:-1]==".jp":
			
			try:
				im = misc.imread("notTrain/"+file)
				images[iii,:,:,:] = imutils.resize(im, width=24)
				labels.append(0)
				iii+=1
				if iii%100==0:
					print iii
			except Exception, e:
				print e
				
images = images[0:9000]
labels = labels[0:9000]				

np.save("poolImagesTrain",images2)	
np.save("poolLabelsTrain",np.array(labels2))
np.save("notPoolImagesTrain",images)
np.save("notPoolLabelsTrain",np.array(labels))