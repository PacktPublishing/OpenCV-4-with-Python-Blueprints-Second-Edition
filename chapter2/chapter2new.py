#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCV with Python Blueprints
    Chapter 2: Hand Gesture Recognition Using a Kinect Depth Sensor

    An app to detect and track simple hand gestures in real-time using the
    output of a Microsoft Kinect 3D Sensor.
"""

import numpy as np

import wx
import cv2
import freenect

from wx_gui import BaseLayout
from gestures import recognize

__author__ = "Michael Beyeler"
__license__ = "GNU GPL 3.0 or later"

def main():
    def process_frame(frame):
        """Recognize hand gesture in a frame of the depth sensor"""
        # clip max depth to 1023, convert to 8-bit grayscale
        np.clip(frame, 0, 2**10 - 1, frame)
        frame >>= 2
        frame = frame.astype(np.uint8)

        # recognize hand gesture
        num_fingers, img_draw = recognize(frame)

        # draw some helpers for correctly placing hand
        height, width = frame.shape[:2]
        cv2.circle(img_draw, (width // 2, height // 2), 3, [255, 102, 0], 2)
        cv2.rectangle(img_draw, (width // 3, height // 3), (width * 2 // 3, height * 2 // 3), (255, 102, 0),2)

        # print number of fingers on image
        cv2.putText(img_draw, str(num_fingers), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        return img_draw
    def frame_iterator():
        frame = freenect.sync_get_depth()[0]
        while frame is not None:
            yield frame
            frame = freenect.sync_get_depth()[0]

    for frame in frame_iterator():
        cv2.imshow("frame",process_frame(frame))
        cv2.waitKey(10)



if __name__ == '__main__':
    main()

# capture = cv2.VideoCapture()
# capture.open(cv2.CAP_OPENNI)
# capture.isOpened()
# frame, _ = freenect.sync_get_depth()
# import matplotlib.pyplot as plt
# plt.hist(frame.ravel(),bins=100)
# plt.hist(frame.ravel()/8,bins=100)
# plt.imshow(frame)
# plt.imshow(np.clip(frame,0,2**10-1).astype(np.uint8))
# plt.imshow(np.clip(frame,0,2**10-1)/4)
# plt.imshow(np.clip(frame/4,0,2**8).astype(np.uint8))
# plt.hist(np.clip(frame,0,1100).ravel().astype(np.uint8),bins=100)
# plt.hist((frame>1046).ravel().astype(np.uint8))
# frame.dtype
# plt.imshow((np.clip(frame,0,1000)/8).astype(np.uint8))
# plt.imshow(frame.astype(np.uint8))
# plt.imshow((frame).astype(np.uint8))
# np.array([257],dtype=np.uint16).astype(np.uint8)
