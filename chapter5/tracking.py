#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module that contains an algorithm for multiple-objects tracking"""

import cv2
import numpy as np
import copy
import itertools

# cv2.

class MultipleObjectsTracker:
    """
    Multiple-objects tracker

    This class implements an algorithm for tracking multiple objects in
    a video sequence.
    The algorithm combines a saliency map for object detection and
    mean-shift tracking for object tracking.
    """

    def __init__(self, min_area=400, min_shift2=1):
        """
        Constructor

        This method initializes the multiple-objects tracking algorithm.

        :param min_area: Minimum area for a proto-object contour to be
                         considered a real object
        :param min_shift2: Minimum distance for a proto-object to drift
                           from frame to frame ot be considered a real
                           object
        """
        self.object_box = []

        self.min_cnt_area = min_area
        self.min_shift2 = min_shift2

        # Setup the termination criteria, either 100 iteration or move by at
        # least 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          5, 1)
        self.rejected = 0
        self.num_tracked = 0

    def advance_frame(self, frame, proto_objects_map,saliency):
        """
        Advance the algorithm by a single frame

        This method tracks all objects via the following steps:
         - adds all bounding boxes from saliency map as potential
           targets
         - finds bounding boxes from previous frame in current frame
           via mean-shift tracking
         - combines the two lists by removing duplicates

        certain targets are discarded:
         - targets that are too small
         - targets that don't move

        :param frame: New input RGB frame
        :param proto_objects_map: corresponding proto-objects map of the
                                  frame
        :returns: frame annotated with bounding boxes around all objects
                  that are being tracked
        """
        print("advanced")
        tracker = copy.deepcopy(frame)
        # Build a list all bounding boxes found from the
        # current proto-objects map
        box_all = self._boxes_from_saliency(proto_objects_map,self.min_cnt_area)

        # find all bounding boxes extrapolated from last frame
        # via mean-shift tracking
        # meanshift_boxes = self._boxes_from_meanshift(frame)
        if len(self.object_box) >= len(box_all) - self.rejected:
            # if len(self.object_box):
            print("box all overwrite")
            box_all = self._boxes_from_meanshift2(saliency)
            self.num_tracked+=1
        else:
            self.rejected = 0
            self.num_tracked = 0
            self.centeres = self.get_centeres(box_all)
        self.object_box = box_all

        # draw remaining boxes
        self.draw_boxes(tracker)
        return tracker
    def draw_boxes(self,frame):
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
        # cv.PutText(img, text, org, font, color) → None¶
        for (x, y, w, h),num in zip(self.object_box,itertools.count()):
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            cv2.putText(frame, str(num),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
            # cv2.putText(frame, str(num), org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    @staticmethod
    def _boxes_from_saliency(proto_objects_map,min_cnt_area):
        """
        :param proto_objects_map: proto-objects map of the current frame
        :param box_all: append bounding boxes from saliency to this list
        :returns: new list of all collected bounding boxes
        """
        # find all bounding boxes in new saliency map
        cnt_sal, _ = cv2.findContours(proto_objects_map, 1, 2)
        sal_boxes =  [cv2.boundingRect(cnt) for cnt in cnt_sal
                if cv2.contourArea(cnt) > min_cnt_area]
        return sal_boxes
    def get_centeres(self,boxes):
        return [(x+w/2,y+h/2) for (x,y,w,h) in boxes]
    def _boxes_from_meanshift2(self, saliency):
        meanshift_boxes = [cv2.meanShift(saliency,tuple(box_old),self.term_crit)[1] for box_old in self.object_box]
        return meanshift_boxes
