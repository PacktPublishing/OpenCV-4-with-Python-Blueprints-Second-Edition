#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for camera calibration using a chessboard"""


import cv2
import numpy as np
import wx

from wx_gui import BaseLayout


class CameraCalibration(BaseLayout):
    """Camera calibration

        Performs camera calibration on a webcam video feed using
        the chessboard approach described here:
        http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    """

    def augment_layout(self):
        pnl = wx.Panel(self, -1)
        self.button_calibrate = wx.Button(pnl, label='Calibrate Camera')
        self.Bind(wx.EVT_BUTTON, self._on_button_calibrate)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.button_calibrate)
        pnl.SetSizer(hbox)

        self.panels_vertical.Add(pnl, flag=wx.EXPAND | wx.BOTTOM | wx.TOP,
                                 border=1)

        # setting chessboard size (size of grid - 1)
        # (7,7) for the standard chessboard
        self.chessboard_size = (7, 7)

        # prepare object points
        self.objp = np.zeros((np.prod(self.chessboard_size), 3),
                             dtype=np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                                    0:self.chessboard_size[1]].T.reshape(-1, 2)

        # prepare recording
        self.recording = False
        self.record_min_num_frames = 15
        self._reset_recording()

    def process_frame(self, frame):
        """Processes each frame

            If recording mode is on (self.recording==True), this method will
            perform all the hard work of the camera calibration process:
            - for every frame, until enough frames have been processed:
                - find the chessboard corners
                - refine the coordinates of the detected corners
            - after enough frames have been processed:
                - estimate the intrinsic camera matrix and distortion
                  coefficients

            :param frame: current RGB video frame
            :returns: annotated video frame showing detected chessboard corners
        """
        # if we are not recording, just display the frame
        if not self.recording:
            return frame

        # else we're recording
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        if self.record_cnt < self.record_min_num_frames:
            # need at least some number of chessboard samples before we can
            # calculate the intrinsic matrix

            ret, corners = cv2.findChessboardCorners(img_gray,
                                                     self.chessboard_size,
                                                     None)
            if ret:
                print(f"{self.record_min_num_frames - self.record_cnt} chessboards remain")
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)

                # refine found corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30, 0.01)
                cv2.cornerSubPix(img_gray, corners, (9, 9), (-1, -1), criteria)

                self.obj_points.append(self.objp)
                self.img_points.append(corners)
                self.record_cnt += 1

        else:
            # we have already collected enough frames, so now we want to
            # calculate the intrinsic camera matrix (K) and the distortion
            # vector (dist)
            print("Calibrating...")
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points,
                                                             self.img_points,
                                                             (self.imgHeight,
                                                              self.imgWidth),
                                                             None, None)
            print("K=", K)
            print("dist=", dist)

            # double-check reconstruction error (should be as close to zero as
            # possible)
            mean_error = 0
            for obj_point, rvec, tvec, img_point in zip(
                    self.obj_points, rvecs, tvecs, self.img_points):
                img_points2, _ = cv2.projectPoints(
                    obj_point, rvec, tvec, K, dist)
                error = cv2.norm(img_point, img_points2,
                                 cv2.NORM_L2) / len(img_points2)
                mean_error += error

            print("mean error=", mean_error)

            self.recording = False
            self._reset_recording()
            self.button_calibrate.Enable()
        return frame

    def _on_button_calibrate(self, event):
        """Enable recording mode upon pushing the button"""
        self.button_calibrate.Disable()
        self.recording = True
        self._reset_recording()

    def _reset_recording(self):
        """Disable recording mode and reset data structures"""
        self.record_cnt = 0
        self.obj_points = []
        self.img_points = []


def main():
    capture = cv2.VideoCapture(0)
    assert capture.isOpened(), "Can not connect to camera"
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = CameraCalibration(capture, title='Camera Calibration', fps=2)
    layout.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
