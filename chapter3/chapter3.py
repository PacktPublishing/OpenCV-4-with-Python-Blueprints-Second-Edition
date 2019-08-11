#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCV with Python Blueprints
    Chapter 3: Finding Objects Via Feature Matching and Perspective Transforms

    An app to detect and track an object of interest in the video stream of a
    webcam, even if the object is viewed at different angles, distances, or
    under partial occlusion.
"""

import cv2 as cv
from feature_matching import FeatureMatching

def main():
    capture = cv.VideoCapture(0)
    assert capture.isOpened(), "Cannot connect to camera"

    capture.set(cv.CAP_PROP_FPS, 5)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    matching = FeatureMatching(train_image='salinger.jpg')

    for success, frame in iter(capture.read, (False, None)):
        cv.imshow("frame", frame)
        match_succsess, img_warped, img_flann = matching.match(frame)
        if match_succsess:
            cv.imshow("res", img_warped)
            cv.imshow("flann", img_flann)
        if cv.waitKey(1) & 0xff == 27:
            break


if __name__ == '__main__':
    main()
