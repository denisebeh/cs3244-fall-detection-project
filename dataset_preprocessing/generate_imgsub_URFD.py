
import os
import cv2
import glob
import sys

#from scipy import ndimage, misc
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
import numpy as np

# import the necessary packages
import argparse
import datetime
import time


### settings ###
data_folder = 'C:/Users/lzyda/Desktop/Uni Readings/Y2S2/CS3244/cs3244-fall-detection-project/URFD_images'  #insert your folder
output_path = 'C:/Users/lzyda/Desktop/Uni Readings/Y2S2/CS3244/cs3244-fall-detection-project/URFD_imgsub_xy'  #insert output folder
thresh_level = 20
################

# init vars

#print("files", glob.glob(data_folder + '/**/',  recursive = True))


class_num = 0

if not os.path.exists(output_path): 
	os.mkdir(output_path)

folders = [f for f in os.listdir(data_folder + '/') if os.path.isdir(os.path.join(data_folder + '/', f))]
folders.sort()
for folder in folders: #iterate through folders fall and non-fall

        event_folders = [f for f in os.listdir(data_folder + '/' + folder) if os.path.isdir(os.path.join(data_folder + '/' + folder + '/', f))]
        event_folders.sort()
        for event_folder in event_folders: #iterate through folders with image

                path = data_folder + '/' + folder + '/' + event_folder
                imagesub = output_path + '/' + folder + '/' + event_folder
                if not os.path.exists(imagesub):
                        os.makedirs(imagesub)
                #print(imagesub, path)

		## image sub algo
                prevFrame_flag = True
                
                for i, file in enumerate([f for f in os.listdir(path)]): #iterate through all images in said folder
                        try:
                                thisFrame = cv2.imread(path + '/' + file)
                                thisFrame_gr = cv2.cvtColor(thisFrame, cv2.COLOR_BGR2GRAY) #grayscale
                                thisFrame_gr_blur = cv2.GaussianBlur(thisFrame_gr, (9, 9), 0) # gaussian blur to reduce noice
                                
                                #print("processed")

                                
                                
                        except:
                                print("file error - skipping")
                                continue


                        #edge case: check if prevFrame is empty. If empty, fill it with currframe and go to next frame
                        if prevFrame_flag:
                                prevFrame = thisFrame
                                prevFrame_gr_blur = thisFrame_gr_blur
                                prevFrame_flag = False
                                continue

                        # compute image diff
                        frameDelta1 = cv2.subtract(prevFrame_gr_blur, thisFrame_gr_blur) #forward diff
                        #frameDelta2 = cv2.GaussianBlur(frameDelta1, (9,9), cv2.BORDER_DEFAULT) #backward diff
                        #frameDelta = cv2.max(0, frameDelta1)
                        #print("subtracted")

                        # set threshold
                        thresh = cv2.threshold(frameDelta1, thresh_level, 255, cv2.THRESH_BINARY)[1]
                        # dilate image to fill in holes
                        thresh = cv2.dilate(thresh, np.ones((3,3), np.uint8), iterations = 1)

                        # write images to directory
                        imsave(imagesub + '/' + str(i) + 'x' + ".jpg", thresh)
                        imsave(imagesub + '/'+ str(i)  + 'y' + ".jpg", thresh)
                        print(file)

                        ## lmk if you want save the prev / gr / delta frames but dk how

                        # next loop prep
                        prevFrame = thisFrame
                        prevFrame_gr_blur = thisFrame_gr_blur
                                
                        print(file)
                        

                








