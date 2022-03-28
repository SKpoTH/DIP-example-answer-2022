import numpy as np
import cv2 as cv

def euclidean(input_img, center):
    '''
        Distance Calcuation - Euclidean Methods
    '''
    y, x, z = input_img.shape

    ### -> Flattern
    positions = input_img.reshape(y*x, z)
    centers = center * np.ones((y*x, 1))

    ### -> Find Euclidean Distances
    distances = np.sqrt(np.sum((positions - centers)**2, axis=1))

    # print(distances)
    # print(distances.min(), distances.max())

    distances = distances.reshape((y, x))

    return distances

def mahalanobis(input_img, center):
    '''
        Distance Calucation - Mahalanobis Methods (Non-modified Mahalanobis)
    '''
    y, x, z = input_img.shape

    # print(input_img)

    ### -> Flattern
    positions = input_img.reshape(y*x, z)
    centers = center * np.ones((y*x, 1))

    # print(positions)

    ### -> Find Invert Covariance
    cov = np.cov(positions.T)
    inv_cov = np.linalg.inv(cov)

    ### -> Find Mahalanobis Distance
    delta = positions - centers
    distances = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))

    distances = distances.reshape((y, x))

    return distances    

def colorSpaceSegment(input_img, color_center, cutoff, distance_type="euclidean"):
    '''
        Color Segmentation - Accept color within given distance cutoff
                           - Reject color outside given distance cutoff
    '''
    distanceFunction = {
                            "euclidean": euclidean(input_img, color_center),
                            "mahalanobis": mahalanobis(input_img, color_center)
                       }

    distance_map = distanceFunction.get(distance_type)

    output_img = np.where(distance_map > cutoff, 0, 255)

    return output_img