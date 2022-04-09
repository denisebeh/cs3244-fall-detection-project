
import os
import cv2
import glob
import sys

from scipy import ndimage, misc
import matplotlib.pyplot as plt
import numpy as np

# import the necessary packages
import argparse
import datetime
import time
import cv2
%matplotlib inline

#sys.path.append('/usr/local/lib/python2.7/site-packages/') #append sys path not needed as imported packages

### settings ###
data_folder = '~/URFD_images'  #insert your folder
output_path = '/URFD_imagesub' #insert output folder
thresh_level = 15
################

# init vars
prevFrame = None
total_files = len(glob.glob(data_folder)) #total num to be processed
movement_graph = [0] * total_files
class_num = 0
print("total files:",total_files)


if not os.path.exists(output_path): 
    os.mkdir(output_path)

folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
folders.sort()
for folder in folders: #iterate through folders fall and non-fall

    event_folders = [f for f in os.listdir(data_folder + folder) if os.path.isdir(os.path.join(data_folder + folder + '/', f))]
    event_folders.sort()
    for event_folder in event_folders: #iterate through folders with image
        
	    path = data_folder + folder + '/' + event_folder
	    imagesub = output_path + folder + '/' + event_folder
	    if not os.path.exists(imagesub):
		os.makedirs(imagesub)
	    print(imagesub, path)

	    ## image sub algo
	    for i,file in enumerate(glob.glob(event_folder)): #iterate through all images in said folder
                try:
                    thisFrame = misc.imread(file)
                    thisFrame_gr = cv2.cvtColor(thisFrame, cv2.COLOR_BGR2GRAY) #grayscale
                    thisFrame_gr_blur = cv2.GaussianBlur(thisFrame.gr, (21, 21), 0) # gaussian blur to reduce noice
                    thisFrame_box = cv2.cvtColor(thisFrame, cv2.COLOR_BGR2GRAY)
                except:
                    "file error - skipping"
                    continue

                #edge case: check if prevFrame is empty. If empty, fill it with currframe and go to next frame
                if prevFrame == None:
                    prevFrame = thisFrame
                    prevFrame_gr_blur = thisFrame_gr_blur
                    continue

                # compute image diff
                frameDelta1 = cv2.subtract(prevFrame_gr_blur, thisFrame_gr_blur) #forward diff
                frameDelta2 = cv2.subtract(thisFrame_gr_blur, prevFrame_gr_blur) #backward diff
                frameDelta = cv2.max(0, frameDelta1)

                # set threshold
                tresh = cv2.threshold(frameDelta, thresh_level, 255, cs2.THRESH_BINARY)[1]
                # dilate image to fill in holes
                tresh = cv2.dilate(tresh, None, iterations = 2)
                # find contours of dilated image
                (cnts, _) = cv2.findContours(tresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # convert to a scalar
                thresh_ones = np.zeros(tresh.shape)
                thresh_ones[thresh==255] = 1
                movement_graph[i] = np.sum(tresh_ones)

                # write images to directory
                print(file)
                misc.imsave(imagesub + file + ".png", thisFrame)
                ## lmk if you want save the prev / gr / delta frames but dk how


                ##### status #######
                sys.stdout.write('\r'+'Processing File: '+str(i)+ ' of '+ str(total_files))

                # next loop prep
                prevFrame = thisFrame
                prevFrame_gr_blur = thisFrame_gr_blur

            # print movement graph
            plt.plot(movement_graph)

            





            
