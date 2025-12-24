import cv2
import os
import numpy as np
from time import time
from PIL import Image

def read_data(folder, grayscale=True):
    '''Read data from the folder

    Parameters
    ----------
    folder: str, the directory with images
    grayscale: bool, apply grayscaling on the RGB or not
    -------
    Returns:
    files: np.ndarray, array with images'''

    # parse images
    path = os.path.join('data', folder, 'img')
    files = next(os.walk(path))[2]
    files = sorted(files)
    # open images
    files = np.array(list(map(lambda x: np.array(Image.open(os.path.join(path, x))), files)))

    # apply or not grayscaling
    if grayscale:
        if len(files.shape) == 4:
            files = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), files)))

    return files

def record_video(rois, video_data, filename, fps=20, mode='rectangle'):
    '''Save the video with frames to the .avi file

    Parameters
    ----------
    rois: list of tuples, the rectantular roi list with
    correspondent frames
    video_data: np.ndarray, 4D array with images (video)
    filename: str, output filename
    fps: int, Frames per second
    mode: str, the mode of function depends on the
    type of roi: rectangular or polygonal points
    -------
    Returns: None, saves the video'''

    # check if the frames grayscale or RGB
    is_rgb = len(video_data.shape) == 4

    if is_rgb:
        shape = video_data[0].shape[:2][::-1]
    else:
        shape = video_data[0].shape[::-1]

    # create object for recording video
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    # create output format
    if filename[:-4] == '.avi':
        video_filename = filename
    else:
        video_filename = '{}.avi'.format(filename)

    # save video depends on the gray or RGB format
    if is_rgb:
        out = cv2.VideoWriter(video_filename, fourcc, fps, shape)
    else:
        out = cv2.VideoWriter(video_filename, fourcc, fps, shape, 0)

    # iterate through frames, add rectangle and save frame
    for roi, frame in zip(rois, video_data):

        tmp = frame.copy()

        # apply mode
        if mode == 'rectangle':
            cv2.rectangle(tmp, (roi[2], roi[0]),
                               (roi[3], roi[1]),
                               (0, 0, 255), 2)
        if mode == 'polygon':

            filler = cv2.convexHull(np.flip(roi, axis=1)).reshape(1, -1, 2)
            cv2.polylines(tmp, filler, True, (0, 255, 255), 2)

        if is_rgb:
            out.write(cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
        else:
            out.write(tmp)

    out.release()
