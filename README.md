This aim of this project is to detect swimming pools in satellite images. A two step process is used.

First, the file detectBlueBlobs.py uses a simple color threshold to find blue objects, and it crops and saves them as jpg images.

Second, the file classify/tenFlow.py trains a 5 layer CNN to sort the images into one of two categories: pool or not pool. It takes about 30 minutes to train on my laptop.

The network can by run by calling checkTenFlow.py. In my most recent experiment I was getting 94% precision with 80% recall. 

The file convertImagesToArrays.py is used to save the training data as a np array file which I found to be faster. 

