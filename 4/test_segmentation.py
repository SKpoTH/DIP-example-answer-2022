import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as skmorph
from glob import glob

from SKpoTH.fourier2D import forwardFFT, showMagnitude, inverseFFT
from SKpoTH.utility import convertRange
from SKpoTH.morphology import CCpicker

def readImages(input_path, color_model):
    '''
        Read Image with convert into desired color model
    '''
    input_files = glob(input_path + '*')
    input_imgs = [cv.cvtColor(cv.imread(x), color_model) for x in input_files]

    return input_imgs

def periordicErase(input_img):
    '''
    '''
    fft_magnitude, fft_phase = forwardFFT(input_img)

    # -> Ban Periodic Noise Pattern
    ban_y = 2
    ban_x = 1
    y_center = input_img.shape[0] // 2
    x_center = input_img.shape[1] // 2
    fft_magnitude[y_center-ban_y:y_center+ban_y, :x_center-ban_x] = 0
    fft_magnitude[y_center-ban_y:y_center+ban_y, x_center+ban_x:] = 0

    # showMagnitude(fft_magnitude)

    output_img = inverseFFT(fft_magnitude, fft_phase)

    output_img = output_img.astype(np.uint8)

    return output_img

def crackEnhancement(input_img):
    '''
        Find Crack Area of the Concrete
    '''
    # -> Periordic Elimination
    pe_img = periordicErase(input_img)

    output_img = pe_img

    return output_img

def crackSegmentation(input_img):
    '''
        Crack on concrete segmentation
    '''
    _, thresh_img = cv.threshold(input_img, 70, 255, cv.THRESH_BINARY)
    output_img = 255-CCpicker(255-thresh_img, 0.0001)

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
INPUT_PATH = "images/crack/input/"
OUTPUT_PATH = "images/crack/output/"

if __name__ == "__main__":

    input_imgs = readImages(INPUT_PATH, cv.COLOR_BGR2GRAY)

    output_imgs = []

    for i in range(len(input_imgs)):
        # i = 1
        input_img = input_imgs[i]

        enhance_img = crackEnhancement(input_img)
        segment_img = crackSegmentation(enhance_img)

        output_img = segment_img
        
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