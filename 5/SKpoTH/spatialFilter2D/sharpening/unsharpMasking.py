import numpy as np
import cv2 as cv

def unsharpMasking(input_img, blurfilter_size, k):
    '''
        Unsharp Masking + High Boosting to Sharpening the image
    '''
    ### -> Smooth/Blur Image by Gaussian Filtering
    # > Create Gaussian Filter
    gauss_filter = cv.getGaussianKernel(blurfilter_size, -1)
    gauss_filter = gauss_filter * gauss_filter.T
    # > Filtering 2D
    blur_img = cv.filter2D(input_img, -1, gauss_filter)

    ### -> Unsharp Masking
    mask_img = input_img - blur_img.astype(float)

    ### -> High Boosting
    sharp_img = input_img + (k * mask_img)

    # -> Rounding intensity
    # > Flooring above 255 to 255
    sharp_img[sharp_img>255] = 255
    # > Ceiling below 0 to 0
    sharp_img[sharp_img<0] = 0

    # -> Covert float-to-uint8
    # sharp_img = sharp_img.astype(np.uint8)

    output_img = sharp_img

    return output_img, mask_img, blur_img