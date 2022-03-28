import cv2 as cv
import numpy as np

def inverseFFT(fft_magnitude, fft_phase):
    '''
    '''
    # -> Shift Quardrant back to Spatial Standard format
    fft_magnitude = np.fft.ifftshift(fft_magnitude)

    # -> Tranform Magintude, Phase back to 2D-Signal form and combine as Complex
    img_real = fft_magnitude * np.cos(fft_phase)
    img_imag = fft_magnitude * np.sin(fft_phase)
    img_complex = img_real + (img_imag * 1j)

    # -> Take 2D-Invert Fourier Transform
    output_img = np.fft.ifft2(img_complex)

    return output_img.real