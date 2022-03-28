import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from SKpoTH.segmentation import colorSpaceSegment
from SKpoTH.morphology import CCpicker, floodFill

def readImages(input_path, color_model):
    '''
        Read Image with convert into desired color model
    '''
    input_files = glob(input_path + '*')
    input_imgs = [cv.cvtColor(cv.imread(x), color_model) for x in input_files]

    return input_imgs

def applesSegmentation(input_img):
    '''
        Find Apple Area
    '''
    # input_img = lightEqualize(input_img)

    color_center = (247, 236, 135)
    segment_img = colorSpaceSegment(input_img, color_center, 85, distance_type="euclidean")

    output_img = CCpicker(segment_img, 0.005)
    output_img = floodFill(output_img)

    # plt.imshow(output_img, cmap="gray")
    # plt.show()
    
    return output_img

def writeImages(input_path, output_path, output_imgs):
    '''
        Write image by using input as their name
    '''
    input_files = glob(input_path + '*')

    for i, f in enumerate(input_files):
        _, filename = os.path.split(f)
        print(filename)
        cv.imwrite(output_path + filename, output_imgs[i])

# ======================= HEAD SCRIPT ===========================
INPUT_PATH = "images/apples/input/"
OUTPUT_PATH = "images/apples/output/"

if __name__ == "__main__":

    input_imgs = readImages(INPUT_PATH, cv.COLOR_BGR2RGB)
    output_imgs = []

    for i in range(len(input_imgs)):
        # i = 1
        input_img = input_imgs[i]
        output_img = applesSegmentation(input_img)

        output_imgs.append(output_img)

    writeImages(INPUT_PATH, OUTPUT_PATH, output_imgs)

    # plt.imshow(input_imgs[0])
    # plt.imshow(gt_imgs[0], cmap="gray")
    # plt.subplot(1, 2, 1)
    # plt.imshow(input_img, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(output_img, cmap="gray")
    # plt.show()
    # print(len(input_imgs))
    
# ======================= TAIL SCRIPT ===========================