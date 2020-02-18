#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module that contains various detectors"""

import cv2
import numpy as np


class FaceDetector:
    """Face Detector

        This class implements a face detection algorithm using a face cascade
        and two eye cascades.
    """

    def __init__(self, *,
                 face_cascade='params/haarcascade_frontalface_default.xml',
                 eye_cascade='params/haarcascade_lefteye_2splits.xml',
                 scale_factor=4):
        # resize images before detection
        self.scale_factor = scale_factor

        # load pre-trained cascades
        self.face_clf = cv2.CascadeClassifier(face_cascade)
        if self.face_clf.empty():
            raise ValueError(f'Could not load face cascade "{face_cascade}"')
        self.eye_clf = cv2.CascadeClassifier(eye_cascade)
        if self.eye_clf.empty():
            raise ValueError(
                f'Could not load eye cascade "{eye_cascade}"')

    def detect_face(self, rgb_img, *, outline=True):
        """Performs face detection

            This method detects faces in an RGB input image.
            The method returns True upon success (else False), draws the
            bounding box of the head onto the input image (frame), and
            extracts the head region (head).

            :param frame: RGB input image
            :returns: success, frame, head
        """
        frameCasc = cv2.cvtColor(cv2.resize(rgb_img, (0, 0),
                                            fx=1.0 / self.scale_factor,
                                            fy=1.0 / self.scale_factor),
                                 cv2.COLOR_RGB2GRAY)
        faces = self.face_clf.detectMultiScale(
            frameCasc,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.CASCADE_SCALE_IMAGE) * self.scale_factor

        # if face is found: extract head region from bounding box
        for (x, y, w, h) in faces:
            if outline:
                cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (100, 255, 0),
                              thickness=2)
            head = cv2.cvtColor(rgb_img[y:y + h, x:x + w],
                                cv2.COLOR_RGB2GRAY)
            return True, rgb_img, head, (x, y)

        return False, rgb_img, None, (None, None)

    def eye_centers(self, head, *, outline=False):
        height, width = head.shape[:2]

        eyes = self.eye_clf.detectMultiScale(head,
                                             scaleFactor=1.1,
                                             minNeighbors=3,
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        if len(eyes) != 2:
            raise RuntimeError(f'Number of eyes {len(eyes)} != 2')
        eye_centers = []
        for x, y, w, h in eyes:
            # find the center of the detected eye region
            eye_centers.append(np.array([x + w / 2, y + h / 2]))
            if outline:
                cv2.rectangle(head, (x, y), (x + w, y + h), (10, 55, 0),
                              thickness=2)
        return eye_centers

    def align_head(self, head):
        """Aligns a head region using affine transformations

            This method preprocesses an extracted head region by rotating
            and scaling it so that the face appears centered and up-right.

            The method returns True on success (else False) and the aligned
            head region (head). Possible reasons for failure are that one or
            both eye detectors fail, maybe due to poor lighting conditions.

            :param head: extracted head region
            :returns: success, head
        """
        # we want the eye to be at 25% of the width, and 20% of the height
        # resulting image should be square (desired_img_width,
        # desired_img_height)
        desired_eye_x = 0.25
        desired_eye_y = 0.2
        desired_img_width = desired_img_height = 200

        try:
            eye_centers = self.eye_centers(head)
        except RuntimeError:
            return False, head

        if eye_centers[0][0] < eye_centers[0][1]:
            left_eye, right_eye = eye_centers
        else:
            right_eye, left_eye = eye_centers

        # scale distance between eyes to desired length
        eye_dist = np.linalg.norm(left_eye - right_eye)
        eyeSizeScale = (1.0 - desired_eye_x * 2) * desired_img_width / eye_dist

        # get rotation matrix
        # get center point between the two eyes and calculate angle
        eye_angle_deg = 180 / np.pi * np.arctan2(right_eye[1] - left_eye[1],
                                                 right_eye[0] - left_eye[0])
        eye_midpoint = (left_eye + right_eye) / 2
        rot_mat = cv2.getRotationMatrix2D(tuple(eye_midpoint), eye_angle_deg,
                                          eyeSizeScale)

        # shift center of the eyes to be centered in the image
        rot_mat[0, 2] += desired_img_width * 0.5 - eye_midpoint[0]
        rot_mat[1, 2] += desired_eye_y * desired_img_height - eye_midpoint[1]

        # warp perspective to make eyes aligned on horizontal line and scaled
        # to right size
        res = cv2.warpAffine(head, rot_mat, (desired_img_width,
                                             desired_img_width))

        # return success
        return True, res
