from __future__ import division
import random
import numpy as np
import tensorflow as tf
import os
from scipy import misc
from matplotlib import pyplot as plt
import imutils
from PIL import Image
import shutil

# **********************************Parameters
numClasses = 2
batchSize = 200     
numKernels = 16
#**********************************************

# Returns batches of 200 images and labels.
batchIndex = 0
def getBatch():#toDistort=False):
	global batchIndex
	bat = np.array(range(batchSize))+batchIndex*batchSize
	bat = bat%len(images)
	batchImages = np.array([images[b] for b in bat])
	#batchLabels = np.array([labels[b] for b in bat])
	batchIndex+=1
	batchImages = (batchImages-np.mean(batchImages))/np.std(batchImages)
	return batchImages

# Builds the tensorflow model. (Is only called Once)
def buildModel(images):
	
  # First convolutional layer
  with tf.variable_scope('conv1') as scope:
    weights0 = tf.Variable(tf.truncated_normal([5, 5, 3, numKernels]))								                                         				                                         									
    convOutput = tf.nn.conv2d(images, weights0, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.1, shape=[numKernels]))
    biasOutput = tf.nn.bias_add(convOutput, biases)
    reluOutput = tf.nn.relu(biasOutput, name=scope.name)

  poolOutput1 = tf.nn.max_pool(reluOutput, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
  normOutput1 = tf.nn.lrn(poolOutput1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

  # Second convolutional Layer
  with tf.variable_scope('conv2') as scope:
    weights1 = tf.Variable(tf.truncated_normal([5, 5, numKernels, numKernels]))
    convOutput = tf.nn.conv2d(normOutput1, weights1, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.1, shape=[numKernels]))
    biasOutput = tf.nn.bias_add(convOutput, biases)
    reluOutput0 = tf.nn.relu(biasOutput, name=scope.name)

  normOutput2 = tf.nn.lrn(reluOutput0, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  poolOutput2 = tf.nn.max_pool(normOutput2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Third Layer
  with tf.variable_scope('local3') as scope:
    reshapeOutput = tf.reshape(poolOutput2, [batchSize, -1])
    weights = tf.Variable(tf.truncated_normal([reshapeOutput.get_shape()[1].value, 400]))
    biases = tf.Variable(tf.constant(0.1, shape=[400]))
    reluOutput1 = tf.nn.relu(tf.matmul(reshapeOutput, weights) + biases)

  # Fourth Layer
  with tf.variable_scope('local4') as scope:
    weights = tf.Variable(tf.truncated_normal([400, 200]))
    biases = tf.Variable(tf.constant(0.1, shape=[200]))
    reluOutput2 = tf.nn.relu(tf.matmul(reluOutput1, weights) + biases)

  # Final Layer
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.Variable(tf.truncated_normal([200, numClasses]))
    biases = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    softmax_activation = tf.add(tf.matmul(reluOutput2, weights), biases) # Change to 2
  return softmax_activation

# Placeholders	
x = tf.placeholder(tf.float32, [batchSize,24,24,3])
activation = buildModel(x)

# Starts the sesson
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
saver.restore(sess, "model.ckpt")


iii = 0
paths = []
images = np.array([])
images = np.zeros((3000,24,24,3))
for root, dir, files in os.walk("unsorted"):
	for file in files:
		if file[-4:-1]==".jp":
			try:
				im = misc.imread(root+"/"+file)
				images[iii,:,:,:] = imutils.resize(im, width=24)
				paths.append(root+"/"+file)
				iii+=1
			except Exception, e:
				print e
				
images = images[0:len(paths)]				
				
print images.shape
print len(paths)				

numberOfBatches = int(len(paths)/batchSize)
	
								
print "Testing..."
guesses = np.array([]).reshape(0,2)

for i in range(numberOfBatches):
	batchImages = getBatch()
	temp = activation.eval(feed_dict={x:batchImages}).reshape(batchSize,2)
	print guesses.shape
	guesses = np.concatenate((guesses,temp))

guesses = [np.argmax(guess) for guess in guesses]


print len(paths)
print len(guesses)


for ii in xrange(len(guesses)):
	if guesses[ii]==0:
		# Not a pool
		newPath = paths[ii].replace("unsorted","notPool")
	else:
		# Pool
		newPath = paths[ii].replace("unsorted","pool")
	shutil.move(paths[ii],newPath)