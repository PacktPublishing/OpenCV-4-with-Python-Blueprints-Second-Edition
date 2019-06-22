#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module that contains an algorithm for multiple-objects tracking"""

import cv2
import numpy as np
import copy
import itertools


class MultipleObjectsTracker:
    """
    Multiple-objects tracker

    This class implements an algorithm for tracking multiple objects in
    a video sequence.
    The algorithm combines a saliency map for object detection and
    mean-shift tracking for object tracking.
    """

    def __init__(self, min_object_area: int = 400,
                 min_speed_per_pix: float = 0.02):
        """
        Constructor

        This method initializes the multiple-objects tracking algorithm.

        :param min_area: Minimum area for a proto-object contour to be
                         considered a real object
        """
        self.object_boxes = []
        self.min_object_area = min_object_area
        self.min_speed_per_pix = min_speed_per_pix
        self.num_frame_tracked = 0
        # Setup the termination criteria, either 100 iteration or move by at
        # least 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          5, 1)

    def advance_frame(self,
                      frame: np.ndarray,
                      proto_objects_map: np.ndarray,
                      saliency: np.ndarray) -> np.ndarray:
        """
        Advance the algorithm by a single frame

        certain targets are discarded:
         - targets that are too small
         - targets that don't move

        :param frame: New input RGB frame
        :param proto_objects_map: corresponding proto-objects map of the
                                  frame
        :param saliency: TODO: EXPLAIN
        :returns: frame annotated with bounding boxes around all objects
                  that are being tracked
        """
        print(f"Objects are tracked for {self.num_frame_tracked} frame")

        # Build a list all bounding boxes found from the
        # current proto-objects map
        object_contours, _ = cv2.findContours(proto_objects_map, 1, 2)
        object_boxes = [cv2.boundingRect(contour)
                        for contour in object_contours
                        if cv2.contourArea(contour) > self.min_object_area]

        if len(self.object_boxes) >= len(object_boxes):
            # Continue tracking with meanshift if number of salient objects
            # didn't increase
            object_boxes = [cv2.meanShift(saliency, box, self.term_crit)[1]
                            for box in self.object_boxes]
            self.num_frame_tracked += 1
        else:
            # Otherwise restart tracking
            self.num_frame_tracked = 0
            self.object_initial_centers = [
                (x + w / 2, y + h / 2) for (x, y, w, h) in object_boxes]

        # Remember current objects
        self.object_boxes = object_boxes

        return self.draw_good_boxes(copy.deepcopy(frame))

    def draw_good_boxes(self, frame: np.ndarray) -> np.ndarray:
        # Find total displacement length for each object
        # and normalize by object size
        displacements = [((x + w / 2 - cx)**2 + (y + w / 2 - cy)**2)**0.5 / w
                         for (x, y, w, h), (cx, cy)
                         in zip(self.object_boxes, self.object_initial_centers)]
        # Draw objects that move and their numbers
        for (x, y, w, h), displacement, i in zip(
                self.object_boxes, displacements, itertools.count()):
            # Draw only those which have some avarage speed
            if displacement / (self.num_frame_tracked + 0.01) > self.min_speed_per_pix:
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                cv2.putText(frame, str(i), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        return frame
