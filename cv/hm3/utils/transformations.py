import cv2
import numpy as np

def warp(params):
    '''Compute affine matrix with input
    list of paramterers

    Parameters
    ----------
    params: np.ndarray or list

    -------
    Returns:
    matrix: np.ndarray
    '''
    matrix = np.array([[1.0 + params[0], 0.0 + params[2], 0.0 + params[4]],
                       [0.0 + params[1], 1.0 + params[3], 0.0 + params[5]]])

    return matrix

def roi_to_points(roi):
    '''Conver from region of interest
    in form of slices in the image, i.e
    x0:x1, y0:y1 into the four corner points
    of the rectangle

    Parameters
    ----------
    roi: tuple, region of interect (rectangle)

    -------
    Returns:
    l: np.ndarray, four corner points
    '''

    # define points
    top_left = np.array([roi[0], roi[2]])
    top_right = np.array([roi[0], roi[3]])
    bottom_left = np.array([roi[1], roi[2]])
    bottom_right = np.array([roi[1], roi[3]])

    # add all to the list
    l = []
    l.append(top_left)
    l.append(top_right)
    l.append(bottom_left)
    l.append(bottom_right)
    l = np.array(l).astype(np.int32)

    return l

def indexes(shape):
    '''Compute indexes within the template

    Parameters
    ----------
    shape: tuple, the shape of the template

    -------
    Returns:
    indexes: np.ndarray, the output indexes within
    the matrix grid, i.e matrix with 2D-elements
    '''
    X, Y = np.meshgrid(range(shape[0]), range(shape[1]))
    indexes = np.array([X.flatten(), Y.flatten()]).transpose().reshape((shape[0],
                                                                        shape[1],
                                                                        2))
    return indexes

def to_rectangle(roi):
    '''Transform region of interest
    in the form of points to rectangular form,
    i.e four corner points -> x0:x1, y0:y1

    Parameters
    ----------
    roi: np.ndarray, list of points

    -------
    Returns:
    result: tuple, the roi in form x0:x1, y0:y1'''

    result = (roi[:, 0].min(), roi[:, 0].max(), roi[:, 1].min(), roi[:, 1].max())
    return result

def affine_2d_transform(image, params):
    '''Inverse affine transform of image
    with correspondent parameters

    Parameters
    ----------
    image: np.ndarray, image
    params: np.ndarray, six parameters of affine transform

    -------
    Returns:
    result: tuple, the roi in form x0:x1, y0:y1'''
    
    rows, cols = image.shape[:2]
    matrix = warp(params)
    result = cv2.warpAffine(image, matrix, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return result

def transform_points(points, params):
    '''Warp points with affine transformation,
    encoded by parameters

    Parameters
    ----------
    points: np.ndarray, points on the image
    params: np.ndarray, six parameters of affine transform

    ----------
    Returns:
    points: np.ndarray, transformed points'''

    points = np.flip(points, axis=1)
    points = list(map(lambda x: warp(params) @ np.hstack((x, 1.0)), points))
    points = np.array(points, dtype=np.int32)
    points = np.flip(points, axis=1)

    return points
