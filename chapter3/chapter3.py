#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCV with Python Blueprints
    Chapter 3: Finding Objects Via Feature Matching and Perspective Transforms

    An app to detect and track an object of interest in the video stream of a
    webcam, even if the object is viewed at different angles, distances, or
    under partial occlusion.
"""

import cv2
from feature_matching import FeatureMatching


def main():
    capture = cv2.VideoCapture(0)
    assert capture.isOpened(), "Cannot connect to camera"

    capture.set(cv2.CAP_PROP_FPS, 5)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    train_img = cv2.imread('train.png', cv2.CV_8UC1)
    matching = FeatureMatching(train_img)

    for success, frame in iter(capture.read, (False, None)):
        cv2.imshow("frame", frame)
        match_succsess, img_warped, img_flann = matching.match(frame)
        if match_succsess:
            cv2.imshow("res", img_warped)
            cv2.imshow("flann", img_flann)
        if cv2.waitKey(1) & 0xff == 27:
            break


if __name__ == '__main__':
    main()
