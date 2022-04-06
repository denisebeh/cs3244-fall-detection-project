import os
import cv2
import numpy as np

def rotate_image_left(img):
    height, width = img.shape[:2]
    # first param: rotate from center, second param: angle, third param: scale
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 20, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

def rotate_image_right(img):
    height, width = img.shape[:2]
    # first param: rotate from center, second param: angle, third param: scale
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 340, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

def zoom_image(img, zoom_factor):
    height, width = img.shape[:2]
    # define new boundaries
    x1 = int(0.5 * width * (1 - 1/zoom_factor))
    x2 = int(width - 0.5 * width * (1 - 1/zoom_factor))
    y1 = int(0.5 * height *(1 - 1/zoom_factor))
    y2 = int(height - 0.5 * height * (1 - 1/zoom_factor))
    
    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)

def increase_brightness(img):
    bright = np.ones(img.shape, dtype="uint8") * 150
    bright_increase_img = cv2.add(img, bright)
    return bright_increase_img

def decrease_brightness(img):
    bright = np.ones(img.shape, dtype="uint8") * 70
    bright_decrease_img = cv2.subtract(img, bright)
    return bright_decrease_img

def flip_image_hori(img):
    flipped_img = cv2.flip(img, 3)
    return flipped_img

def sharpen_image(img):
    kernel = np.array([ [0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]  ])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img

def box_blur_image(img):
    blurred_img = cv2.blur(img, (3, 3))
    blurred_img = cv2.blur(blurred_img, (3, 3))
    return blurred_img

def positive_contrast_image(img):
    brightness = 10
    contrast = 70
    dummy = np.int16(img)
    dummy = dummy * (contrast/127+1) - contrast + brightness
    dummy = np.clip(dummy, 0, 255)
    img = np.uint8(dummy)
    return img

def negative_saturation_image(img):
    value = -40
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value >= 0:
        lim = 255 - value
        s[s > lim] = 255
        s[s <= lim] += value
    else:
        lim = np.absolute(value)
        s[s < lim] = 0
        s[s >= lim] -= np.absolute(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

falls_directory = "C:/Users/denisebeh/Downloads/datasets/Falls"
adls_directory = "C:/Users/denisebeh/Downloads/datasets/ADLs"
mode = 0o666

# process falls
for num in range(1,31):
    num = str(num).zfill(2)
    directory1 = 'fall-' + num + '-cam0-rgb'
    path1 = os.path.join(falls_directory, directory1)
    os.mkdir(path1, mode)
    x = len(os.listdir('C:/Users/denisebeh/Downloads/URFD/fall/fall-' + num + '-cam0-rgb')) + 1
    for j in range(1, x):
        data = 'fall-' + num +'-cam0-rgb-' + str(j).zfill(3)
        directory = "fall[" + data + "]"    # sub folder for each image
        parent_dir = path1
        path = os.path.join(parent_dir, directory)
        os.mkdir(path, mode)

        # get augmented images
        input = cv2.imread('C:/Users/denisebeh/Downloads/URFD/fall/fall-' + num + '-cam0-rgb/' + data + '.png')
        image_0 = rotate_image_right(input)
        image_0 = box_blur_image(image_0)
        image_1 = flip_image_hori(input)
        image_1 = decrease_brightness(image_1)
        image_2 = rotate_image_left(input)
        image_2 = positive_contrast_image(image_2)
        image_3 = negative_saturation_image(input)
        os.path.join(path, "")
        cv2.imwrite(path + "/fall0.png", image_0)
        cv2.imwrite(path + "/fall1.png", image_1)
        cv2.imwrite(path + "/fall2.png", image_2)
        cv2.imwrite(path + "/fall3.png", image_3)

# process ADL
for num in range(1, 41):
    num = str(num).zfill(2)
    directory2 = 'adl-' + num + '-cam0-rgb'
    path2 = os.path.join(adls_directory, directory2)
    os.mkdir(path2, mode)
    x = len(os.listdir('C:/Users/denisebeh/Downloads/URFD/adl/adl-' + num + '-cam0-rgb')) + 1
    for j in range(1, x):
        data = 'adl-' + num +'-cam0-rgb-' + str(j).zfill(3)
        directory = "adl[" + data + "]"     # sub folder for each image
        parent_dir = path2
        path = os.path.join(parent_dir, directory)
        os.mkdir(path, mode)
        
        # get augmented images
        input = cv2.imread('C:/Users/denisebeh/Downloads/URFD/adl/adl-' + num + '-cam0-rgb/' + data + '.png')
        image_0 = rotate_image_right(input)
        image_0 = box_blur_image(image_0)
        image_1 = flip_image_hori(input)
        image_1 = decrease_brightness(image_1)
        image_2 = rotate_image_left(input)
        image_2 = positive_contrast_image(image_2)
        image_3 = negative_saturation_image(input)
        os.path.join(path, "")
        cv2.imwrite(path + "/adl0.png", image_0)
        cv2.imwrite(path + "/adl1.png", image_1)
        cv2.imwrite(path + "/adl2.png", image_2)
        cv2.imwrite(path + "/adl3.png", image_3)