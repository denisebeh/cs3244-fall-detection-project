"""
Script used to prepare the URFD dataset: 

1- It takes the downloaded RGB images of the camera 0, divided into two
   folders: 'Falls' and 'ADLs', where all the images of a video (comprised in a
   folder) are inside one of those folders.

2- It creates another folder for the new dataset with the 'Falls' and
   'NotFalls' folders. All the ADL videos are moved to the 'NotFalls' folder.
   The images within the original 'Falls' folder are divided in three stages:
   (i) the pre-fall ADL images (they go to the new 'NotFalls' folder),
   (ii) the fall itself (goes to the new 'Falls' folder) and
   (iii) the post-fall ADL images (to 'NotFalls').
   
All the images are resized to size (W,H) - both W and H are variables of the
script.

The script should generate all the necessary folders.
"""

import cv2
import os
import csv
import sys
import glob
import numpy as np
import zipfile

# Path where the images are stored
# downloads_folder = '/home/user/Downloads/'
data_folder = '/Users/denisebeh/NUSy3s2/cs3244/URFD_images_not_segmented/original_images/'
augmented_data_folder = '/Users/denisebeh/NUSy3s2/cs3244/URFD_images_not_segmented/augmented_v2/'
adl_folder = 'ADLs/'
fall_folder = 'Falls/'
# Path to save the images
output_path = '/Users/denisebeh/NUSy3s2/cs3244/preprocessed_dataset/with_augmented/'
# Label files, download them from the dataset's site
falls_labels = '/Users/denisebeh/NUSy3s2/cs3244/URFD_images_not_segmented/urfall-cam0-falls.csv'
notfalls_labels = '/Users/denisebeh/NUSy3s2/cs3244/URFD_images_not_segmented/urfall-cam0-adls.csv'
W, H = 224, 224 # shape of new images (resize is applied)

# =====================================================================
# UNZIP THE DATASET
# =====================================================================

# if not os.path.exists(data_folder):
#     os.makedirs(data_folder + fall_folder)
#     os.makedirs(data_folder + adl_folder)

# if not os.path.exists(augmented_data_folder):
#     os.makedirs(augmented_data_folder + fall_folder)
#     os.makedirs(augmented_data_folder + adl_folder)

# adl_zipped_files = glob.glob(downloads_folder + 'adl-*-cam0-rgb.zip')
# fall_zipped_files = glob.glob(downloads_folder + 'fall-*-cam0-rgb.zip')

# content = [
#     [adl_zipped_files, data_folder + adl_folder],
#     [fall_zipped_files, data_folder + fall_folder]
# ]
# for zipped_files, dst_folder in content:
#     for zipped_file in zipped_files:
#         zfile = zipfile.ZipFile(zipped_file)
#         zfile.extractall(dst_folder)

# =====================================================================
# READ LABELS AND STORE THEM
# =====================================================================

labels = {'falls': dict(), 'notfalls': dict()}

# For falls videos: read the CSV where frame-level labels are given
with open(falls_labels, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    event_type = 'falls'
    for row in spamreader:
        elems = row[0].split(',')  # read a line in the csv 
        if not elems[0] in labels[event_type]:
            labels[event_type][elems[0]] = []
        if int(elems[2]) == 1 or int(elems[2]) == -1:
            labels[event_type][elems[0]].append(0)
        elif int(elems[2]) == 0:
            labels[event_type][elems[0]].append(1)

# For ADL videos: read the CSV where frame-level labels are given
with open(notfalls_labels, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    event_type ='notfalls'
    for row in spamreader:
        elems = row[0].split(',')  # read a line in the csv 
        if not elems[0] in labels[event_type]:
            labels[event_type][elems[0]] = []
        if int(elems[2]) == 1 or int(elems[2]) == -1:
            labels[event_type][elems[0]].append(0)
        elif int(elems[2]) == 0:
            labels[event_type][elems[0]].append(1)
            
print('Label files processed')
            
# =====================================================================
# PROCESS THE DATASET
# =====================================================================

if not os.path.exists(output_path):
    os.makedirs(output_path + 'Falls')
    os.makedirs(output_path + 'NotFalls')

# Get all folders: each one contains the set of images of the video - Falls or ADLs
folders = [f for f in os.listdir(data_folder)
             if os.path.isdir(os.path.join(data_folder, f))]

for folder in folders: # folder = ADLs or Falls
    print('{} videos =============='.format(folder))
    print(os.listdir(data_folder + folder))
    events = [f for f in os.listdir(data_folder + folder) # each event is a folder eg. adl-01-cam0-rgb
                if os.path.isdir(os.path.join(data_folder + folder, f))]
    events.sort() 
    for nb_event, event, in enumerate(events):
        # Create the appropriate folder
        if folder == 'ADLs':
            event_id = event[:6] # eg. adl-01
            #new_folder = output_path + 'NotFalls/notfall_{}'.format(event_id)
            new_folder = output_path + 'NotFalls/{}'.format(event)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)     
        elif folder == 'Falls':
            event_id = event[:7]
            # "No falls" come before and after the fall, so the respective
            # folders must be created
            #new_folder = output_path + 'Falls/fall_{}'.format(event_id)
            new_folder = output_path + 'Falls/{}'.format(event)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)    
        #folder_created = False

        # ---------------------------------------
        path_to_images = data_folder + folder + '/' + event + '/'

        # Load all the images of the video
        images = [f for f in os.listdir(path_to_images) 
                    if os.path.isfile(os.path.join(path_to_images, f))]
        images.sort()
        fall_detected = False # whether a fall has been detected in the video
        for nb_image, image in enumerate(images):

            # original image processing pipeline
            x = cv2.imread(path_to_images + image)
            
            # If the image is part of an ADL video no fall need to be considered
            if folder == 'ADLs':

                # map to corresponding folder in augmented images
                path_to_augmented = augmented_data_folder + folder + '/' + event + '/adl[' + image[:-4] + ']/'
                
                aug_images = [f for f in os.listdir(path_to_augmented) 
                                if os.path.isfile(os.path.join(path_to_augmented, f))]

                if not os.path.exists(output_path + 'NotFalls/{}'.format(event) + '/frame{}'.format(nb_image)):
                        os.makedirs(output_path + 'NotFalls/{}'.format(event) + '/frame{}'.format(nb_image))
                    
                # Save original image
                save_path = (output_path +
                    #'NotFalls/notfall_{}'.format(event_id) + 
                    'NotFalls/{}'.format(event) + 
                    '/frame{}'.format(nb_image) +
                    '/frame_1.jpg')

                cv2.imwrite(save_path,
                            cv2.resize(x, (W,H)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95]) 
                    
                # Save aug image
                for idx, img in enumerate(aug_images):
                    aug_x = cv2.imread(path_to_augmented + img)

                    save_path = (output_path +
                        #'NotFalls/notfall_{}'.format(event_id) + 
                        'NotFalls/{}'.format(event) + 
                        '/frame{}'.format(nb_image) +
                        '/frame_{}.jpg'.format(idx+2))

                    cv2.imwrite(save_path,
                                cv2.resize(aug_x, (W,H)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])


            elif folder == 'Falls':

                # map to corresponding folder in augmented images
                path_to_augmented = augmented_data_folder + folder + '/' + event + '/fall[' + image[:-4] + ']/'
                aug_images = [f for f in os.listdir(path_to_augmented) 
                                if os.path.isfile(os.path.join(path_to_augmented, f))]
                    
                event_type = 'falls'

                if not os.path.exists(output_path + 'Falls/{}'.format(event) + '/frame{}'.format(nb_image)):
                        os.makedirs(output_path + 'Falls/{}'.format(event) + '/frame{}'.format(nb_image))
                
                if labels[event_type][event_id][nb_image] == 0: # ADL
                    if fall_detected:
                        # Create another folder for an ADL event,
                        # i.e. the post-fall ADL event
                        new_folder = (output_path +
                                    #'NotFalls/notfall_{}_post'.format(
                                    #event_id))
                                    'NotFalls/{}_post'.format(event) +
                                    '/frame{}'.format(nb_image))
                        
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder) 
                        
                        # save original image
                        save_path = (new_folder +
                                    '/frame_1.jpg')
                        
                        cv2.imwrite(save_path,
                                    cv2.resize(x, (W,H)),
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                        # save aug images
                        for idx, img in enumerate(aug_images):
                            aug_x = cv2.imread(path_to_augmented + img)

                            save_path = (new_folder +
                                        '/frame_{}.jpg'.format(idx+2))

                            cv2.imwrite(save_path,
                                        cv2.resize(aug_x, (W,H)),
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    else:
                        new_folder = (output_path +
                                    #'NotFalls/notfall_{}_pre'.format(event_id))
                                    'NotFalls/{}_pre'.format(event) + 
                                    '/frame{}'.format(nb_image))

                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder) 

                        save_path = (new_folder +
                                    '/frame_1.jpg')
                        cv2.imwrite(save_path,
                                    cv2.resize(x, (W,H)),
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                        # save aug images
                        for idx, img in enumerate(aug_images):
                            aug_x = cv2.imread(path_to_augmented + img)

                            save_path = (new_folder +
                                        '/frame_{}.jpg'.format(idx+2))

                            cv2.imwrite(save_path,
                                        cv2.resize(aug_x, (W,H)),
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            
                elif labels[event_type][event_id][nb_image] == 1: # actual fall
                    print("checkpoint 1 ..............")

                    # save original images
                    save_path = (output_path + 
                                #'Falls/fall_{}'.format(event_id) +
                                'Falls/{}'.format(event) +
                                '/frame{}'.format(nb_image) +
                                '/frame_1.jpg')
                    print("save_path: " + save_path)
                    cv2.imwrite(save_path,
                                cv2.resize(x, (W,H)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    # save aug images
                    for idx, img in enumerate(aug_images):
                        print("aug_images : " + str(img))
                        aug_x = cv2.imread(path_to_augmented + img)

                        save_path = (output_path + 
                                    #'Falls/fall_{}'.format(event_id) +
                                    'Falls/{}'.format(event) +
                                    '/frame{}'.format(nb_image) + 
                                    '/frame_{}.jpg'.format(idx+2))

                        print("next save_path : " + save_path )

                        cv2.imwrite(save_path,
                                    cv2.resize(aug_x, (W,H)),
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    # If fall is detected in a video set the variable to True
                    # used to discern between pre- and post-fall ADL events
                    fall_detected = True

print('End of the process, all the images stored within the {} folder'.format(
      output_path))
