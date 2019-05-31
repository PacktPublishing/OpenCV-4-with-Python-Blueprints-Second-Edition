#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A module containing simple GUI layouts using wxPython

This file is heavily based on the work of Michael Beyeler.
"""

__license__ = "GNU GPL 3.0 or later"

import numpy as np
import wx
import cv2


class BaseLayout(wx.Frame):
    """ Abstract base class for all layouts in the book.

    A custom layout needs to implement the 2 methods below
        - augment_layout
        - process_frame
    """

    def __init__(self,
                 capture: cv2.VideoCapture,
                 title: str = None,
                 parent=None,
                 window_id: int = -1,  # default value
                 fps: int = 10):
        """
        Initialize all necessary parameters and generate a basic GUI layout
        that can then be augmented using `self.augment_layout`.

        :param parent: A wx.Frame parent (often Null). If it is non-Null,
            the frame will be minimized when its parent is minimized and
            restored when it is restored.
        :param window_id: The window identifier.
        :param title: The caption to be displayed on the frame's title bar.
        :param capture: Original video source to get the frames from.
        :param fps: Frames per second at which to display camera feed.
        """
        # Make sure the capture device could be set up
        self.capture = capture
        success, frame = self._acquire_frame()
        if not success:
            print("Could not acquire frame from camera.")
            raise SystemExit()
        self.imgHeight, self.imgWidth = frame.shape[:2]

        super().__init__(parent, window_id, title,
                         size=(self.imgWidth, self.imgHeight + 20))
        self.fps = fps
        self.bmp = wx.Bitmap.FromBuffer(self.imgWidth, self.imgHeight, frame)

        # set up periodic screen capture
        self.timer = wx.Timer(self)
        self.timer.Start(1000. / self.fps)
        self.Bind(wx.EVT_TIMER, self._on_next_frame)

        # set up video stream
        self.video_pnl = wx.Panel(self, size=(self.imgWidth, self.imgHeight))
        self.video_pnl.SetBackgroundColour(wx.BLACK)
        self.video_pnl.Bind(wx.EVT_PAINT, self._on_paint)

        # display the button layout beneath the video stream
        self.panels_vertical = wx.BoxSizer(wx.VERTICAL)
        self.panels_vertical.Add(self.video_pnl, 1, flag=wx.EXPAND | wx.TOP,
                                 border=1)

        self.augment_layout()

        # round off the layout by expanding and centering
        self.SetMinSize((self.imgWidth, self.imgHeight))
        self.SetSizer(self.panels_vertical)
        self.Centre()

    def augment_layout(self):
        """ Augment custom layout elements to the GUI.

        This method is called in the class constructor, after initializing
        common parameters. Every GUI contains the camera feed in the variable
        `self.video_pnl`. Additional layout elements can be added below
        the camera feed by means of the method `self.panels_vertical.Add`
        """
        raise NotImplementedError()

    def _on_next_frame(self, event):
        """
        Capture a new frame from the capture device,
        send an RGB version to `self.process_frame`, refresh.
        """
        success, frame = self._acquire_frame()
        if success:
            # process current frame
            frame = self.process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # update buffer and paint (EVT_PAINT triggered by Refresh)
            self.bmp.CopyFromBuffer(frame)
            self.Refresh(eraseBackground=False)

    def _on_paint(self, event):
        """ Draw the camera frame stored in `self.bmp` onto `self.video_pnl`.
        """
        wx.BufferedPaintDC(self.video_pnl).DrawBitmap(self.bmp, 0, 0)

    def _acquire_frame(self) -> (bool, np.ndarray):
        """ Capture a new frame from the input device

        :return: (success, frame)
            Whether acquiring was successful and current frame.
        """
        return self.capture.read()

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Process the frame of the camera (or other capture device)

        :param frame_rgb: Image to process in rgb format, of shape (H, W, 3)
        :return: Processed image in rgb format, of shape (H, W, 3)
        """
        raise NotImplementedError()
