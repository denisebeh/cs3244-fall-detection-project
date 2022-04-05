
import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import os
cam = '1' # change cam no here
for a in range(1,31):
    k = format(a, '02') # change dataset no
    directory1 = 'fall-' + k + '-cam' + cam + '-rgb' # main folder for datafile
    parent_dir1 = "C:/Unu_Stuff/Y2S2/CS3244/Project/datasets"
    mode = 0o666
    path1 = os.path.join(parent_dir1, directory1)
    os.mkdir(path1, mode)
    x = len(os.listdir('C:/Unu_Stuff/Y2S2/CS3244/Project/images/fall-' + k + '-cam' + cam + '-rgb')) + 1 
    for j in range(1, x):
        data = 'fall-' + k +'-cam' + cam + '-rgb-' + format(j, '03')
        directory = "fall[" + data + "]" #sub folder for each image
        parent_dir = path1
        mode = 0o666
        path = os.path.join(parent_dir, directory)
        os.mkdir(path, mode)

        figure = cv2.imread('images/fall-' + k + '-cam' + cam + '-rgb/' + data + '.png')
        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        #view_transform(figure)
        for i in range(0,3):
            transform = A.Compose([A.HorizontalFlip(p=0.5),
                                   A.CLAHE(),
                                   A.Blur(p = 0.2),
                                   A.ColorJitter(),# similar to torchvision
                                   #A.Downscale(p = 0.2), #not always wanna low qual image
                                   #A.RandomShadow(),
                                   A.Affine(rotate = [-20, 20], shear = 0),
                                   A.RGBShift(p = 0.3),
                                   A.HueSaturationValue(sat_shift_limit = 10, p=0.2),
                                   A.RandomBrightnessContrast(
                                       brightness_limit=0.2,
                                       p  = 1 )]) #change ltr
            augmented_image = transform(image=figure)['image']
        #view_transform(augmented_image)

            cv2.imwrite('datasets/' + directory1 + "/" + directory + '/fall' + str(i) + '.png', augmented_image)
        #break
