import numpy as np
import cv2 as cv

def CCpicker(input_img, limit_percent):
    '''
        Pick objects that larger enough
    '''
    _, labels_img = cv.connectedComponents(input_img.astype(np.uint8))

    groups, counts = np.unique(labels_img, return_counts=True)

    target_size = (labels_img.shape[0] * labels_img.shape[1]) * limit_percent
    target_groups = np.argwhere(counts > target_size)[1:]

    output_img = np.isin(labels_img, target_groups.flatten())

    output_img = output_img * 255

    return output_img