#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCV with Python Blueprints
    Chapter 1: Fun with Filters

    An app to apply three different image filter effects to the video stream
    of a webcam in real-time.
    The three effects are:
    * Black-and-white pencil sketch
    * Warming/cooling filters
    * Cartoonizer
"""

import wx
import cv2

from wx_gui import BaseLayout
from tools import apply_hue_filter
from tools import apply_rgb_filters
from tools import load_img_resized
from tools import spline_to_lookup_table
from tools import cartoonize
from tools import convert_to_pencil_sketch
import numpy as np


__license__ = "GNU GPL 3.0 or later"


class FilterLayout(BaseLayout):
    """
    Custom layout for filter effects

    This class implements a custom layout for applying diverse filter
    effects to a camera feed. The layout is based on an abstract base
    class BaseLayout. It displays the camera feed (passed to the class as
    a cv2.VideoCapture object) in the variable self.panels_vertical.
    Additional layout elements can be added by using the Add method (e.g.,
    self.panels_vertical(wx.Panel(self, -1))).
    """
    INCREASE_LOOKUP_TABLE = spline_to_lookup_table([0, 64, 128, 192, 256],
                                                   [0, 70, 140, 210, 256])
    DECREASE_LOOKUP_TABLE = spline_to_lookup_table([0, 64, 128, 192, 256],
                                                   [0, 30, 80, 120, 192])

    def augment_layout(self):
        """ Add a row of radio buttons below the camera feed. """

        # create a horizontal layout with all filter modes as radio buttons
        pnl = wx.Panel(self, -1)
        self.mode_warm = wx.RadioButton(pnl, -1, 'Warming Filter', (10, 10),
                                        style=wx.RB_GROUP)
        self.mode_cool = wx.RadioButton(pnl, -1, 'Cooling Filter', (10, 10))
        self.mode_sketch = wx.RadioButton(pnl, -1, 'Pencil Sketch', (10, 10))
        self.mode_cartoon = wx.RadioButton(pnl, -1, 'Cartoon', (10, 10))
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.mode_warm, 1)
        hbox.Add(self.mode_cool, 1)
        hbox.Add(self.mode_sketch, 1)
        hbox.Add(self.mode_cartoon, 1)
        pnl.SetSizer(hbox)

        # add panel with radio buttons to existing panels in a vertical
        # arrangement
        self.panels_vertical.Add(pnl, flag=wx.EXPAND | wx.BOTTOM | wx.TOP,
                                 border=1)

    def _render_warm(self, rgb_image: np.ndarray) -> np.ndarray:
        interim_img = apply_rgb_filters(rgb_image,
                                        red_filter=self.INCREASE_LOOKUP_TABLE,
                                        blue_filter=self.DECREASE_LOOKUP_TABLE)
        return apply_hue_filter(interim_img, self.INCREASE_LOOKUP_TABLE)

    def _render_cool(self, rgb_image: np.ndarray) -> np.ndarray:
        interim_img = apply_rgb_filters(rgb_image,
                                        red_filter=self.DECREASE_LOOKUP_TABLE,
                                        blue_filter=self.INCREASE_LOOKUP_TABLE)
        return apply_hue_filter(interim_img, self.DECREASE_LOOKUP_TABLE)

    def _render_pencil_sketch(self, rgb_image: np.ndarray) -> np.ndarray:
        sketch = convert_to_pencil_sketch(rgb_image)

        canvas = load_img_resized('pencilsketch_bg.jpg',
                                  (self.imgWidth, self.imgHeight))
        if canvas is not None:
            sketch = cv2.multiply(sketch, canvas, scale=1. / 256)

        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Process the frame of the camera (or other capture device)

        Choose a filter effect based on the which of the radio buttons
        was clicked.

        :param frame_rgb: Image to process in rgb format, of shape (H, W, 3)
        :return: Processed image in rgb format, of shape (H, W, 3)
        """
        if self.mode_warm.GetValue():
            return self._render_warm(frame_rgb)
        elif self.mode_cool.GetValue():
            return self._render_cool(frame_rgb)
        elif self.mode_sketch.GetValue():
            return self._render_pencil_sketch(frame_rgb)
        elif self.mode_cartoon.GetValue():
            return cartoonize(frame_rgb)
        else:
            raise NotImplementedError()


def main():
    # open webcam
    capture = cv2.VideoCapture(0)
    if not(capture.isOpened()):
        capture.open()

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = FilterLayout(capture, title='Fun with Filters')
    layout.Center()
    layout.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
