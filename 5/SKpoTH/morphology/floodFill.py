import numpy as np
import cv2 as cv

def floodFill(input_img):
    '''
        Fill holes
    '''
    fill_img = input_img.copy()

    # -> Flood Fill
    mask = np.zeros((fill_img.shape[0]+2, fill_img.shape[1]+2), np.uint8)
    cv.floodFill(fill_img, mask, (0, 0), 255)
    fill_img = 255 - fill_img

    # -> Merge Original and Filled Area
    output_img = input_img | fill_img

    return output_img