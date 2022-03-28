import numpy as np
import cv2 as cv

def boxFiltering(input_img, filter_size):
    '''
        Smoothing Image by Box/Average Filter
    '''
    # -> Create Box/Average Filter
    box_filter = np.ones((filter_size, filter_size))
    box_filter = (1/(filter_size*filter_size)) * box_filter

    # -> Filtering 2D
    filterd_img = cv.filter2D(input_img, -1, box_filter)

    output_img = filterd_img

    return output_img