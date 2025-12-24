from PIL import Image
import numpy as np
import cv2
from time import time, sleep
from pynput.mouse import Controller
import argparse

class CalmanFilter(object):
    '''CalmanFilter class with Calman Filter without conrol'''
    def __init__(self, waiting_time=0.1):

        self.waiting_time = waiting_time
        self.shape = (900, 1440, 3)
        '''State Transition matrix'''
        self.A = np.array([[1.0, 0.0, self.waiting_time, 0.0],
                           [0.0, 1.0, 0.0, self.waiting_time],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        '''Measurement matrix'''
        self.H = np.array([[1.0, 0.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0]])
        '''Action Uncertainty matrix'''
        self.Q = np.array([[0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.1, 0.0],
                           [0.0, 0.0, 0.0, 0.1]])

        #Sensor Noise matrix
        self.R = 0.1 * np.eye(4)

        self.P = np.zeros((4, 4))

        # object for estimation the position of mouse
        self.mouse = Controller()

        # initial prediction
        mouse_x, mouse_y = self.mouse.position
        self.x = np.array([mouse_x, mouse_y, 0, 0])

        self.measurements = []
        self.predictions = []

    def measure(self):
        # measure position and estimate velocity
        position_x, position_y = self.mouse.position

        try:
            velocity_x = (position_x - self.predictions[-1][0]) / self.waiting_time
            velocity_y = (position_y - self.predictions[-1][1]) / self.waiting_time
        except IndexError:
            velocity_x = 0
            velocity_y = 0

        self.measurements.append(np.array([position_x,
                                          position_y,
                                          velocity_x,
                                          velocity_y]))

    def predict(self):
        # prediction stage
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.transpose() + self.Q

    def correct(self):
        # correction stage
        S = self.H @ self.P @ self.H.transpose() + self.R
        K = self.P @ self.H.transpose() @ np.linalg.inv(S)
        y = self.measurements[-1] - self.H @ self.x

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.predictions.append(self.x)

    def mouse_tracking(self, release_time, output_file=None):
        # start tracking
        start = time()
        image = np.zeros(self.shape, np.uint8)

        if output_file != None:
            fps = 20
            # create object for recording video
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(output_file, fourcc, fps, self.shape[:2][::-1])

        while True:
            # wait some time for discretization of measurements
            sleep(self.waiting_time)

            self.measure()
            self.predict()
            self.correct()

            real_values = tuple(self.measurements[-1][:2].astype(np.int32))

            cv2.circle(image, real_values, 3, (255, 0, 0), -1)
            if len(self.predictions) > 2:

                predicted_value_second = tuple(self.predictions[-2][:2].astype(np.int32))
                predicted_value_first = tuple(self.predictions[-1][:2].astype(np.int32))

                cv2.line(image, predicted_value_second,
                                predicted_value_first, (0, 255, 0), 3)
            if output_file != None:
                frame = image.copy()
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            cv2.imshow("Mouse tracking", image)
            k = cv2.waitKey(10) & 0XFF

            if time() - start > release_time:
                break

        if output_file != None:
            out.release()
        cv2.destroyAllWindows()

        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=60)
    parser.add_argument("--output_video", type=str, default='output.avi')
    args = parser.parse_args()

    # run mouse tracker
    obj = CalmanFilter()
    obj.mouse_tracking(args.t, args.output_video)
