#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module containing an algorithm for hand gesture recognition"""

import numpy as np
import cv2
from typing import Tuple

__author__ = "Michael Beyeler"
__license__ = "GNU GPL 3.0 or later"

def recognize(img_gray):
    """Recognizes hand gesture in a single-channel depth image

        This method estimates the number of extended fingers based on
        a single-channel depth image showing a hand and arm region.
        :param img_gray: single-channel depth image
        :returns: (num_fingers, img_draw) The estimated number of
                   extended fingers and an annotated RGB image
    """

    # segment arm region
    segment = segment_arm(img_gray)

    # find the hull of the segmented area, and based on that find the
    # convexity defects
    (contour, defects) = find_hull_defects(segment)

    # detect the number of fingers depending on the contours and convexity
    # defects, then draw defects that belong to fingers green, others red
    img_draw = cv2.cvtColor(segment, cv2.COLOR_GRAY2RGB)
    (num_fingers, img_draw) = detect_num_fingers(contour,
                                                 defects, img_draw)

    return (num_fingers, img_draw)


def segment_arm(frame: np.ndarray, abs_depth_dev: int = 14) -> np.ndarray:
    """Segments arm region

        This method accepts a single-channel depth image of an arm and
        hand region and extracts the segmented arm region.
        It is assumed that the hand is placed in the center of the image.
        :param frame: single-channel depth image
        :returns: binary image (mask) of segmented arm region, where
                  arm=255, else=0
    """
    height, width = frame.shape
    # find center (21x21 pixel) region of imageheight frame
    center_half = 10  # half-width of 21 is 21/2-1
    center = frame[height // 2 - center_half:height // 2 + center_half,
                   width // 2 - center_half:width // 2 + center_half]

    # find median depth value of center region
    med_val = np.median(center)

    # try this instead:
    frame = np.where(abs(frame - med_val) <= abs_depth_dev,
                     128, 0).astype(np.uint8)

    # morphological
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    # connected component
    small_kernel = 3
    frame[height // 2 - small_kernel:height // 2 + small_kernel,
          width // 2 - small_kernel:width // 2 + small_kernel] = 128

    mask = np.zeros((height + 2, width + 2), np.uint8)
    flood = frame.copy()
    cv2.floodFill(flood, mask, (width // 2, height // 2), 255,
                  flags=4 | (255 << 8))

    ret, flooded = cv2.threshold(flood, 129, 255, cv2.THRESH_BINARY)
    return flooded


def find_hull_defects(segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find hull defects

        This method finds all defects in the hull of a segmented arm
        region.
        :param segment: a binary image (mask) of a segmented arm region,
                        where arm=255, else=0
        :returns: (max_contour, defects) the largest contour in the image
                  and all corresponding defects
    """
    contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # find largest area contour
    max_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(max_contour, True)
    max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

    # find convexity hull and defects
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)

    return max_contour, defects



def detect_num_fingers(contour: np.ndarray, defects: np.ndarray,
                       img_draw: np.ndarray, thresh_deg: float = 80.0) -> Tuple[int, np.ndarray]:
    """Detects the number of extended fingers

        This method determines the number of extended fingers based on a
        contour and convexity defects.
        It will annotate an RGB color image of the segmented arm region
        with all relevant defect points and the hull.
        :param contours: a list of contours
        :param defects: a list of convexity defects
        :param img_draw: an RGB color image to be annotated
        :returns: (num_fingers, img_draw) the estimated number of extended
                  fingers and an annotated RGB color image
    """

    # if there are no convexity defects, possibly no hull found or no
    # fingers extended
    if defects is None:
        return [0, img_draw]

    # we assume the wrist will generate two convexity defects (one on each
    # side), so if there are no additional defect points, there are no
    # fingers extended
    if len(defects) <= 2:
        return [0, img_draw]

    # if there is a sufficient amount of convexity defects, we will find a
    # defect point between two fingers so to get the number of fingers,
    # start counting at 1
    num_fingers = 1
    # Defects are of shape (num_defects,1,4)
    for defect in defects[:, 0, :]:
        # Each defect is an array of four integers.
        # First three indexes of start, end and the furthest
        # points respectively
        # contour is of shape (num_points,1,2) - 2 for point coordinates
        start, end, far = [contour[i][0] for i in defect[:3]]
        # draw the hull
        cv2.line(img_draw, tuple(start), tuple(end), (0, 255, 0), 2)

        # if angle is below a threshold, defect point belongs to two
        # extended fingers
        if angle_rad(start - far, end - far) < deg2rad(thresh_deg):
            # increment number of fingers
            num_fingers += 1

            # draw point as green
            cv2.circle(img_draw, tuple(far), 5, (0, 255, 0), -1)
        else:
            # draw point as red
            cv2.circle(img_draw, tuple(far), 5, (0, 0, 255), -1)

    # make sure we cap the number of fingers
    return min(5, num_fingers), img_draw


def angle_rad(v1, v2):
    """Angle in radians between two vectors

        This method returns the angle (in radians) between two array-like
        vectors using the cross-product method, which is more accurate for
        small angles than the dot-product-acos method.
    """
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))




def deg2rad(angle_deg):
    """Convert degrees to radians

        This method converts an angle in radians e[0,2*np.pi) into degrees
        e[0,360)
    """
    return angle_deg / 180.0 * np.pi
