import numpy as np
import imutils
import cv2
from scipy import misc
from matplotlib import pyplot as plt
from PIL import Image
import os

# satelliteLA is a directory containing raw satellite images from the LA area
for root, dirs, files in os.walk("satelliteLA"):
	for myFile in files:
		if myFile == ".DS_Store":
			continue
		print root + "/" + myFile	
		frame = misc.imread(root + "/" +myFile)
		print frame.shape
		frame = imutils.resize(frame, width=500)
		
		# Convert to HSV color space
		img_hsv=cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
		lower_blue = np.array([70,35,80])
		upper_blue = np.array([230,255,255])
		mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
	
		
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)
	
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
	
		frame2=cv2.copyMakeBorder(frame, top=500, bottom=500, left=500, right=500, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
		frame2 = frame
		if len(cnts) > 0:
			for c in cnts:
				((x, y), radius) = cv2.minEnclosingCircle(c)
				if radius > 2:
					#frame1 = frame2[y-radius+500:y+radius+500,x-radius+500:x+radius+500]
					frame1 = frame2[y-radius:y+radius,x-radius:x+radius]
					try:
						frame1 = imutils.resize(frame1, width=32) 
					
						F = Image.fromarray(frame1)
						s = str(np.random.rand(1))
						F.save("classify/unsorted"+s+".jpg")
					except:
						continue
	
				
		