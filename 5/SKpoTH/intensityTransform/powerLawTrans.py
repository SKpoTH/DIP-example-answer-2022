import numpy as np

def powerLawTrans(input_img, gamma):
    '''
        Power Law Intensity Transformation
    '''
    # -> Convert uint8-to-float
    input_img = input_img.astype(float)

    ### -> Power Law Intensity Transformation
    norm_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    output_img = (norm_img ** gamma) * 255

    # -> Convert float-to-uint8
    output_img = output_img.astype(np.uint8)

    return output_img