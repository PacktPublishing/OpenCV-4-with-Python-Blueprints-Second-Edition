#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCV with Python Blueprints
    Chapter 3: Finding Objects Via Feature Matching and Perspective Transforms

    An app to detect and track an object of interest in the video stream of a
    webcam, even if the object is viewed at different angles, distances, or
    under partial occlusion.
"""

import cv2
import wx

from wx_gui import BaseLayout
from feature_matching import FeatureMatching


class FeatureMatchingLayout(BaseLayout):
    """A custom layout for feature matching display

        A plain GUI layout for feature matching output.
        Each captured frame is passed to the FeatureMatching class, so that an
        object of interest can be tracked.
    """

    def augment_layout(self):
        """Initializes feature matching class"""
        self.matching = FeatureMatching(train_image='salinger.jpg')
        self.to_show = None
    def process_frame(self, frame):
        """Processes each captured frame"""
        # if object detected, display new frame, else old one
        success, new_frame = self.matching.match(frame)
        if success:
            self.to_show = new_frame
        # return new_frame
        if self.to_show is not None:
            return self.to_show
        else:
            return frame
        return self.to_show or frame
        return new_frame if success else frame


def mainold():
    capture = cv2.VideoCapture(0)
    if not(capture.isOpened()):
        capture.open()

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = FeatureMatchingLayout(capture, title='Feature Matching')
    layout.Show(True)
    app.MainLoop()

def main():
    capture = cv2.VideoCapture(0)
    assert capture.isOpened(), "Cannot connect to camera"

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    matching = FeatureMatching(train_image='salinger.jpg')

    for success,frame in iter(capture.read,(False,None)):
        cv2.imshow("frame",frame)

        suc,img_out, img_flann = matching.match(frame)

        if suc:
            cv2.imshow("res", img_out)
            cv2.imshow("flann", img_flann)
        k = cv2.waitKey(1) & 0xff


if __name__ == '__main__':
    main()
