from __future__ import division
import random
import numpy as np
import tensorflow as tf
import os
from scipy import misc
from matplotlib import pyplot as plt
import imutils
from PIL import Image


# **********************************Parameters
numClasses = 2
batchSize = 200  
initialLearningRate = 0.005     
numKernels = 64
learningCurve = []
distortTrainingImages = True
decayRate = False   # Causes learning rate to decrease
iterations = 500
restore = True
restoreRate = .001
#**********************************************
if restore:
	initialLearningRate = restoreRate
	iterations = 100

# Randomly Scale and Rotate Images
def distort(image,toDistort):
	if not toDistort:
		return image
	else:		
		scaleFactor = random.randint(24,26)
		image = misc.imresize(image,(scaleFactor,scaleFactor))
		
		#rotFactor = random.randint(-1,1)*3.14159/180
		#image = misc.imrotate(image,rotFactor)
		
		x = random.randint(0,scaleFactor - 24)
		y = random.randint(0,scaleFactor - 24)
		
		image = image[x:24+x,y:24+y,:]
		return image

# Returns batches of 200 images and labels.
batchIndex = 0
def getBatch(toDistort=False):
	global batchIndex
	bat = np.array(range(batchSize))+batchIndex*batchSize
	bat = bat%len(labels)
	batchImages = np.array([distort(images[b],toDistort) for b in bat])
	batchLabels = np.array([labels[b] for b in bat])
	batchIndex+=1
	batchImages = (batchImages-np.mean(batchImages))/np.std(batchImages)
	return batchImages, batchLabels


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

  # Compute cross entropy Loss
def loss(activation, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      activation, labels)
  cross_entropy_mean = tf.reduce_mean(cross_entropy)
  return cross_entropy_mean

  # Returns accuracy
def accuracy(activation, labels):
	prediction = tf.nn.softmax(activation)
	labels = tf.cast(labels,tf.int64)
	correctPrediction = tf.equal(tf.argmax(prediction,1),labels)
	return tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
	

# Placeholders	
x = tf.placeholder(tf.float32, [batchSize,24,24,3])
y = tf.placeholder(tf.float32, [batchSize])


activation = buildModel(x)
loss_ = loss(activation,y)
accuracy_ = accuracy(activation,y)

# Defines Trainer
if decayRate:
	step = tf.Variable(0, trainable=False)
	rate = tf.train.exponential_decay(initialLearningRate, step, 1, 0.999)
	trainer = tf.train.AdamOptimizer(rate).minimize(loss_, global_step=step)
else:
	trainer = tf.train.AdamOptimizer(initialLearningRate).minimize(loss_)


# Loads in training images and Labels
images = np.load('data/poolImagesTrain.npy')
labels = np.load('data/poolLabelsTrain.npy')
images = np.concatenate((images,np.load('data/notPoolImagesTrain.npy')),axis=0)
labels = np.hstack((labels,np.load('data/notPoolLabelsTrain.npy')))
xValues = range(len(labels))
random.shuffle(xValues)
images = np.array([images[i] for i in xValues])
labels = np.array([labels[i] for i in xValues])

# Starts the sesson
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()



if restore:
	saver.restore(sess, "model2.ckpt")

for i in xrange(iterations):

	batchImages, batchLabels = getBatch(distortTrainingImages)
	if i%2==0:
		accu = accuracy_.eval(feed_dict={x:batchImages, y: batchLabels})
		learningCurve.append(1-accu) 
		print accu
	trainer.run(feed_dict={x:batchImages, y: batchLabels})
save_path = saver.save(sess, "model2.ckpt")


batchIndex = 0
# Loads in the test data
images = np.load('data/poolImagesTest.npy')
labels = np.load('data/poolLabelsTest.npy')
images = np.concatenate((images,np.load('data/notPoolImagesTest.npy')),axis=0)
labels = np.hstack((labels,np.load('data/notPoolLabelsTest.npy')))


print "Testing..."
desiredResponses = np.array([]).reshape(0,1)
guesses = np.array([]).reshape(0,2)
accs = []
for i in range(2):
	batchImages, batchLabels = getBatch(False)
	acc = accuracy_.eval(feed_dict={x:batchImages, y: batchLabels})
	accs.append(acc)
	temp =  y.eval(feed_dict={x:batchImages, y: batchLabels}).reshape(200,1)
	temp2 = activation.eval(feed_dict={x:batchImages, y: batchLabels}).reshape(200,2)
	desiredResponses = np.concatenate((desiredResponses,temp))
	guesses = np.concatenate((guesses,temp2))

print "Final Acc:"
print np.mean(np.array(accs))


guesses = [np.argmax(guess) for guess in guesses]

#print guesses
#exit()
# Print confusion matrix
x = np.zeros((2,2))
for ii in xrange(len(guesses)):
	x[int(guesses[ii])][int(desiredResponses[ii])] += 1
print (x)

t = np.arange(len(learningCurve))*10
plt.plot(t,learningCurve)
plt.xlabel("Iteration")
plt.ylabel("Training Error Rate")
plt.show()

