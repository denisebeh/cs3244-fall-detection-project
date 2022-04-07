import os
import cv2
import glob
import time

data_folder = '/Users/denisebeh/NUSy3s2/cs3244/preprocessed_dataset/with_augmented/'
images = ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg", "frame_4.jpg", "frame_5.jpg"]

folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
folders.sort()
for folder in folders:
    #if actor_folder != 'Ahmad': continue
    event_folders = [f for f in os.listdir(data_folder + folder) if os.path.isdir(os.path.join(data_folder + folder + '/', f))]
    event_folders.sort()

    for event_folder in event_folders:
        print('event folder :' + event_folder)
        path = data_folder + folder + '/' + event_folder +'/'
        frames = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        frames.sort()
        # process all the different types of augmented images 
        for image in images:
            img_array = []
            for frame in frames:
                print("image, frame: " + image + ',' + frame)
                current_directory = os.path.join(path, frame)
                image_path = os.path.join(current_directory, image)
                img = cv2.imread(image_path)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
            out = cv2.VideoWriter(path + image[:-4] + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
            
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()