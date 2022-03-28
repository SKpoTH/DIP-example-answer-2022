import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def showMagnitude(fft_magnitude, method="LOG-SCALE", ban_size=0):
    '''
        Magnitude Visualization
    '''
    if method == "LOG-SCALE":
        # => LOG Transformation
        show_magnitude = np.log(1 + fft_magnitude)
        
        plt.imshow(show_magnitude, cmap="gray")
        plt.show()

    elif method == "CENTER-BAN":
        # => Center Banning
        show_magnitude = fft_magnitude
        # Create Center Ban
        center_ban = np.ones_like(show_magnitude)
        center_ban = np.ascontiguousarray(center_ban)
        # Define "ban_size"
        if ban_size <= 0:
            ban_size = min([center_ban.shape[0], center_ban.shape[1]])
            ban_size = int(ban_size * 0.02)
        # Draw Circle
        center = (center_ban.shape[1]//2, center_ban.shape[0]//2)
        center_ban = cv.circle(center_ban, center, ban_size, 0, -1)
        # Ban Center
        show_magnitude = show_magnitude * center_ban

        plt.imshow(show_magnitude, cmap="jet")
        plt.show()