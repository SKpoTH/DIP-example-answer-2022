import numpy as np

def convertRange(input_img, min_val, max_val):
    '''
        Convert current range into give range
    '''
    # -> Convert to [0, 1]
    norm_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    # -> Convert to [min_val, max_val]
    output_img = (norm_img * (max_val - min_val)) + min_val

    return output_img