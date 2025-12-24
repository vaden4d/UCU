import cv2
import os
import numpy as np
from time import time
import argparse
import sys

from utils.data_manipulation import read_data, record_video
from utils.metrics import ssd, sad, ncc

def template_match_loc(image, template, template_func):
    '''Finding the best match via convolving the template
    across the input image.

    Parameters
    ----------
    image: np.ndarray, image
    template: np.ndarray, template for tracking
    template_func: metric for similarity between
    part of image and template

    -------
    Returns:
    max_loc: tuple, the location of the best
    match
    '''

    image = image.astype(float)
    template = template.astype(float)

    image_shape = np.array(image.shape)
    template_shape = np.array(template.shape)

    height, width = image_shape - template_shape + 1

    output = np.zeros((height, width))

    # convolve within image with template as kernel
    for i in range(height):
        for j in range(width):

            output[i][j] = template_func(image[i:i+template_shape[0],
                                               j:j+template_shape[1]], template)
    # compute location of the best match
    _, _, _, max_loc = cv2.minMaxLoc(output)

    return max_loc

def template_matching(video, init_roi, template_func, verbose=True):
    '''Vanilla implementation of template matching.

    Parameters
    ----------
    video: np.ndarray, sequence of images for tracking
    init_roi: tuple, the roi of the object in the rectangular
    form, i.e x0:x1, y0:y1
    template_func: metric for similarity between
    part of image and template
    verbose: bool, detailed training output or not

    -------
    Returns:
    rois: np.ndarray, the list of rois in the
    rectangular form, i.e x0:x1, y0:y1
    '''
    rois = [list(init_roi)]

    all_time = 0

    for i in range(1, len(video)):

        start = time()

        roi = rois[-1].copy()

        old_frame = video[i-1]
        new_frame = video[i]

        template = old_frame[roi[0]:roi[1],
                             roi[2]:roi[3]]

        max_loc = template_match_loc(new_frame, template, template_func)

        roi[0] = max_loc[1]
        roi[1] = max_loc[1] + template.shape[0]
        roi[2] = max_loc[0]
        roi[3] = max_loc[0] + template.shape[1]

        rois.append(roi)

        delta_t = time() - start
        all_time += delta_t

        if verbose:
            print('{} iteration: {:.4f} seconds'.format(i, delta_t))

    print('Template matching algorithm: ')
    print('Overall computation time - {:.4f} minutes'.format(all_time / 60.0))

    return np.array(rois)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", nargs='+', type=int)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    roi = tuple(args.roi)

    try:
        func = locals()[args.metric]
    except KeyError:
        print('Error! Such template metric function does not exist.')
        sys.exit()

    data = read_data(args.data, grayscale=True)
    data_rgb = read_data(args.data, grayscale=False)

    rois = template_matching(data, roi, func, verbose=args.verbose)
    record_video(rois, data_rgb, args.output_video)
