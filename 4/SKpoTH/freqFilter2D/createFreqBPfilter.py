import numpy as np
import cv2 as cv

def idealFunction(distance, freq_band, freq_width):
    '''
        Apply Ideal Function (Ordinary Circle)
    '''
    freq_filter = distance.copy()
    freq_filter = np.where((distance > (freq_band-freq_width/2)) & 
                           (distance < (freq_band+freq_width/2)), 1, 0)

    return freq_filter

def gaussianFunction(distance, freq_band, freq_width):
    '''
        Apply Gaussian Function
    '''
    freq_filter = np.exp(-((distance**2-freq_band**2) / (distance*freq_width))**2)

    return freq_filter

def butterworthFunction(distance, freq_band, freq_width, n_order):
    '''
        Apply Butterworth Function
    '''
    freq_filter = 1 / (1 + ((distance**2-freq_band**2) / (distance*freq_width))**(2*n_order))

    return freq_filter

def createFreqBPfilter(filter_shape, freq_band, freq_width, freq_pos, filter_type="IDEAL", n_order=1):
    '''
        Create Frequency Domain Band-Pass Filter (Ring Shape)
    '''
    IMG_HEIGH, IMG_WIDTH = filter_shape

    ### -> Create Distance Map
    y = np.arange((1-IMG_HEIGH%2)*0.5, IMG_HEIGH+(1-IMG_HEIGH%2)*0.5)
    x = np.arange((1-IMG_WIDTH%2)*0.5, IMG_WIDTH+(1-IMG_WIDTH%2)*0.5)
    xv, yv = np.meshgrid(x, y)
    # Find Distance Map
    distance = ((yv - freq_pos[0])**2 + (xv - freq_pos[1])**2)**0.5

    ### -> Apply Chosen Methods
    freq_filter = {
                    "IDEAL" : idealFunction(distance, freq_band, freq_width),
                    "GAUSSIAN" : gaussianFunction(distance, freq_band, freq_width),
                    "BUTTERWORTH" : butterworthFunction(distance, freq_band, freq_width, n_order),
                  }[filter_type]

    return freq_filter