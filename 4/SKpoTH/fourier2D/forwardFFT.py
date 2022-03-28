import cv2 as cv
import numpy as np

def forwardFFT(input_img):
    '''
    '''
    ### -> Take 2D-Fourier Transform
    fft_complex = np.fft.fft2(input_img)

    ### -> Find Magnitude, Phase consequently
    # > Magnitude
    fft_magnitude = np.abs(fft_complex)
    # > Phase
    fft_phase = np.arctan2(fft_complex.imag, fft_complex.real)

    # -> Shift Magnitude Quardrant
    fft_magnitude = np.fft.fftshift(fft_magnitude)

    return fft_magnitude, fft_phase