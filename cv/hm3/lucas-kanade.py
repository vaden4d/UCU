import cv2
import os
import numpy as np
from time import time
import argparse
import sys

from utils.data_manipulation import read_data, record_video
from utils.transformations import warp, roi_to_points,\
                                 indexes, to_rectangle,\
                                 affine_2d_transform, transform_points

def lucas_kanade(video, initial_roi, eps, verbose=False):
    '''Lucas-Kanade algorithm
    implementation with affine warping
    transformation between frames

    Parameters
    ----------
    video: np.ndarray, video for tracking
    initial_roi: tuple, the roi of the object
    in the form x0:x1, y0:y1
    eps: float, the stopping criteria threshold (norm of
    the parameters increment)
    verbose: bool, the variable for
    extended log of traning

    -------
    Returns:
    rois: np.ndarray, the rois of the
    object across video in the points
    format
    '''
    rois = [roi_to_points(initial_roi)]

    # accumulator of computation time
    all_time = 0

    for i in range(1, len(video)):

        start = time()

        old_frame = video[i-1]
        new_frame = video[i]

        iters = 1
        params = np.zeros(6)

        while True:

            # unpack the current roi
            roi = to_rectangle(rois[-1])

            # cut template
            template = old_frame[roi[0]:roi[1],
                                 roi[2]:roi[3]]

            # warp the destination image with transformation
            warped = affine_2d_transform(new_frame, params)
            warped = warped[roi[0]:roi[1],
                            roi[2]:roi[3]]

            # compute error
            error = np.subtract(template, warped, dtype=np.float64)

            # warp the gradient of image with transformation
            gradient_x = cv2.Sobel(new_frame, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(new_frame, cv2.CV_32F, 0, 1, ksize=3)

            gradient_x = gradient_x[roi[0]:roi[1],
                                    roi[2]:roi[3]]
            gradient_y = gradient_y[roi[0]:roi[1],
                                    roi[2]:roi[3]]

            warped_gradient_x = affine_2d_transform(gradient_x, params)
            warped_gradient_y = affine_2d_transform(gradient_y, params)

            warped_gradient = np.dstack((warped_gradient_x,
                                         warped_gradient_y))

            # compute indexes for jacobian
            template_indexes = indexes(template.shape)

            # compute jacobian
            jacobian_matrix = np.concatenate((
                            np.einsum('ij,km->ijkm', template_indexes[..., 0].astype(np.int32), np.eye(2)),
                            np.einsum('ij,km->ijkm', template_indexes[..., 1].astype(np.int32), np.eye(2)),
                            np.einsum('ij,km->ijkm', np.ones(template.shape), np.eye(2))), axis=3)

            # compute steepest images
            steepest_images = np.einsum('ijk,ijkm->ijm', warped_gradient, jacobian_matrix)

            # compute hessian and its inverse
            hessian = np.einsum('ijk,ijt->ijkt', steepest_images, steepest_images).sum(axis=(0, 1))
            inv_hessian = np.linalg.inv(hessian)

            # compute delta_p
            delta_p = inv_hessian @ np.einsum('ijk,ij->ijk', steepest_images, error).sum(axis=(0, 1))

            if np.linalg.norm(delta_p) < eps:
                break

            if iters > 50:
                break

            params += delta_p
            iters += 1

        rois.append(transform_points(rois[-1], params))

        delta_t = time() - start
        all_time += delta_t
        if verbose:
            print('{} iteration: {:.4f} seconds'.format(i, delta_t))

    print('Lucas-Kanade algorithm: ')
    print('Overall computation time - {:.4f} minutes'.format(all_time / 60.0))

    return rois

def lucas_kanade_with_pyramids(video, initial_roi, eps=1e-1, n_deep=2, verbose=False):
    '''Lucas-Kanade algorithm
    implementation with affine warping
    transformation between frames and
    with pyramidal extension

    Parameters
    ----------
    video: np.ndarray, video for tracking
    initial_roi: tuple, the roi of the object
    in the form x0:x1, y0:y1
    eps: float, the stopping criteria threshold (norm of
    the parameters increment)
    n_deep: int, number of resizing in the pyramid
    verbose: bool, the variable for
    extended log of traning

    -------
    Returns:
    rois: np.ndarray, the rois of the
    object across video in the points
    format
    '''
    rois = [roi_to_points(initial_roi)]

    # accumulator of computation time
    all_time = 0

    for i in range(1, len(video)):

        start = time()

        old_frame = video[i-1]
        new_frame = video[i]


        params = np.zeros(6)

        for j in reversed(range(n_deep)):
            iters = 1
            # scaling coefficient
            coeff = 2**j
            while True:

                # unpack the current roi
                roi = to_rectangle(rois[-1])
                roi = list(map(lambda x: x // coeff, roi))

                old_frame_resized = cv2.resize(old_frame,
                                                dsize=(old_frame.shape[1] // coeff,
                                                       old_frame.shape[0] // coeff))

                new_frame_resized = cv2.resize(new_frame,
                                                dsize=(new_frame.shape[1] // coeff,
                                                       new_frame.shape[0] // coeff))

                # cut template
                template = old_frame_resized[roi[0]:roi[1],
                                             roi[2]:roi[3]]

                # warp the destination image with transformation
                warped = affine_2d_transform(new_frame_resized, params)
                warped = warped[roi[0]:roi[1],
                                roi[2]:roi[3]]

                # compute error
                error = np.subtract(template, warped, dtype=np.float64)

                # warp the gradient of image with transformation
                gradient_x = cv2.Sobel(new_frame_resized, cv2.CV_32F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(new_frame_resized, cv2.CV_32F, 0, 1, ksize=3)

                gradient_x = gradient_x[roi[0]:roi[1],
                                        roi[2]:roi[3]]
                gradient_y = gradient_y[roi[0]:roi[1],
                                        roi[2]:roi[3]]

                warped_gradient_x = affine_2d_transform(gradient_x, params)
                warped_gradient_y = affine_2d_transform(gradient_y, params)

                warped_gradient = np.dstack((warped_gradient_x,
                                             warped_gradient_y))

                # compute indexes for jacobian
                template_indexes = indexes(template.shape)

                # compute jacobian
                jacobian_matrix = np.concatenate((
                                np.einsum('ij,km->ijkm', template_indexes[..., 0].astype(np.int32), np.eye(2)),
                                np.einsum('ij,km->ijkm', template_indexes[..., 1].astype(np.int32), np.eye(2)),
                                np.einsum('ij,km->ijkm', np.ones(template.shape), np.eye(2))), axis=3)

                # compute steepest images
                steepest_images = np.einsum('ijk,ijkm->ijm', warped_gradient, jacobian_matrix)

                # compute hessian and its inverse
                hessian = np.einsum('ijk,ijt->ijkt', steepest_images, steepest_images).sum(axis=(0, 1))
                inv_hessian = np.linalg.inv(hessian)

                # compute delta_p
                delta_p = inv_hessian @ np.einsum('ijk,ij->ijk', steepest_images, error).sum(axis=(0, 1))

                if np.linalg.norm(delta_p) < eps:
                    break

                if iters > 50:
                    break

                params += delta_p
                iters += 1

        rois.append(transform_points(rois[-1], params))

        delta_t = time() - start
        all_time += delta_t
        if verbose:
            print('{} iteration: {:.4f} seconds'.format(i, delta_t))

    print('Lucas-Kanade algorithm with pyramidal extension: ')
    print('Overall computation time - {:.4f} minutes'.format(all_time / 60.0))

    return rois

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", nargs='+', type=int)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    roi = tuple(args.roi)

    data = read_data(args.data, grayscale=True)
    data_rgb = read_data(args.data, grayscale=False)

    rois = lucas_kanade(data, roi, eps=args.eps, verbose=args.verbose)
    record_video(rois, data_rgb, args.output_video, mode='polygon')
