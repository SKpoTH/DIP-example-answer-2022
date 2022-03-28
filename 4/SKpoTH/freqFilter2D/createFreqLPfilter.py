import numpy as np
import cv2 as cv

def idealFunction(distance, freq_cutoff):
    '''
        Apply Ideal Function (Ordinary Circle)
    '''
    freq_filter = distance.copy()
    freq_filter[distance > freq_cutoff] = 0
    freq_filter[distance <= freq_cutoff] = 1

    return freq_filter

def gaussianFunction(distance, freq_cutoff):
    '''
        Apply Gaussian Function
    '''
    freq_filter = np.exp(-distance**2 / (2*freq_cutoff**2))

    return freq_filter

def butterworthFunction(distance, freq_cutoff, n_order):
    '''
        Apply Butterworth Function
    '''
    freq_filter = 1 / (1 + (distance/freq_cutoff)**(2*n_order))

    return freq_filter

def createFreqLPfilter(filter_shape, freq_cutoff, freq_pos, filter_type="IDEAL", n_order=1):
    '''
        Create Frequency Domain Low-Pass Filter
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
                    "IDEAL" : idealFunction(distance, freq_cutoff),
                    "GAUSSIAN" : gaussianFunction(distance, freq_cutoff),
                    "BUTTERWORTH" : butterworthFunction(distance, freq_cutoff, n_order),
                  }[filter_type]

    return freq_filter