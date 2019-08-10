#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module containing an algorithm for feature matching"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Sequence

__author__ = "Michael Beyeler"
__license__ = "GNU GPL 3.0 or later"

cv2.cornerHarris


class Outlier(Exception):
    pass


class FeatureMatching:
    """Feature matching class

        This class implements an algorithm for feature matching and tracking.

        A SURF descriptor is obtained from a training or template image
        (train_image) that shows the object of interest from the front and
        upright.

        The algorithm will then search for this object in every image frame
        passed to the method FeatureMatching.match. The matching is performed
        with a FLANN based matcher.

        Note: If you want to use this code (including SURF) in a non-commercial
        application, you will need to acquire a SURF license.
    """

    def __init__(self, train_image: str = "salinger.jpg") -> None:
        """
        Initialize the SURF descriptor, FLANN matcher, and the tracking
        algorithm.

        :param train_image: training or template image showing the object
        of interest
        """
        # initialize SURF
        self.f_extractor = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
        # template image: "train" image
        # later on compared ot each video frame: "query" image
        self.img_obj = cv2.imread(train_image, cv2.CV_8UC1)
        assert self.img_obj is not None, f"Could not find train image {train_image}"

        self.sh_train = self.img_obj.shape[:2]
        self.key_train, self.desc_train = \
            self.f_extractor.detectAndCompute(self.img_obj, None)

        # initialize FLANN
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # self.flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        # initialize tracking
        self.last_hinv = np.zeros((3, 3))
        self.max_error_hinv = 50.
        self.num_frames_no_success = 0
        self.max_frames_no_success = 5

    def match(self,
              frame: np.ndarray) -> Tuple[bool,
                                          Optional[np.ndarray],
                                          Optional[np.ndarray]]:
        """Detects and tracks an object of interest in a video frame

            This method detects and tracks an object of interest (of which a
            SURF descriptor was obtained upon initialization) in a video frame.
            Correspondence is established with a FLANN based matcher.

            The algorithm then applies a perspective transform on the frame in
            order to project the object of interest to the frontal plane.

            Outlier rejection is applied to improve the tracking of the object
            from frame to frame.

            :param frame: input (query) image in which to detect the object
            :returns: (success, frame) whether the detection was successful and
                      and the perspective-transformed frame
        """

        # create a working copy (grayscale) of the frame
        # and store its shape for convenience
        img_query = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sh_query = img_query.shape  # rows,cols

        # --- feature extraction
        # detect keypoints in the query image (video frame)
        # using SURF descriptor
        key_query, desc_query = self.f_extractor.detectAndCompute(
            img_query, None)
        # img_keypoints = cv2.drawKeypoints(img_query, key_query, None,
        #      (255, 0, 0), 4)
        # cv2.imshow("keypoints",img_keypoints)
        # --- feature matching
        # returns a list of good matches using FLANN
        # based on a scene and its feature descriptor
        good_matches = self._match_features(desc_query)

        try:
            # early outlier detection and rejection
            if len(good_matches) < 4:
                raise Outlier("Too few matches")

            # --- corner point detection
            # calculates the homography matrix needed to convert between
            # keypoints from the train image and the query image
            dst_corners = self._detect_corner_points(key_query, good_matches)
            # early outlier detection and rejection
            # if any corners lie significantly outside the image, skip frame
            if np.any((dst_corners < -20) | (dst_corners > sh_query)):
                raise Outlier("Out of image")
            # early outlier detection and rejection
            # find the area of the quadrilateral that the four corner points
            # spans
            area = 0
            for prev, nxt in zip(dst_corners, np.roll(
                    dst_corners, -1, axis=0)):
                area += (prev[0] * nxt[1] - prev[1] * nxt[0]) / 2.

            # early outlier detection and rejection
            # reject corner points if area is unreasonable
            if not np.prod(sh_query) / 16. < area < np.prod(sh_query) / 2.:
                raise Outlier("Area is unreasonably small or large")
            # outline corner points of train image in query image
            img_flann = draw_good_matches(
                self.img_obj,
                self.key_train,
                img_query,
                key_query,
                good_matches)
            # adjust x-coordinate (col) of corner points so that they can be drawn
            # next to the train image (add self.sh_train[1])
            dst_corners[:, 0] += self.sh_train[1]
            cv2.polylines(
                img_flann,
                [dst_corners.astype(np.int)],
                isClosed=True,
                color=(
                    0,
                    255,
                    0),
                thickness=3)

            # --- bring object of interest to frontal plane
            Hinv, dst_size = self._warp_keypoints(good_matches, key_query,
                                                  sh_query)

            # outlier rejection
            # if last frame recent: new Hinv must be similar to last one
            # else: accept whatever Hinv is found at this point
            recent = self.num_frames_no_success < self.max_frames_no_success
            similar = np.linalg.norm(
                Hinv - self.last_hinv) < self.max_error_hinv
            if recent and not similar:
                raise Outlier("Not similar transformation")
        except Outlier as e:
            print(f"Outlier:{e}")
            self.num_frames_no_success += 1
            return False, None, None
        else:
            # reset counters and update Hinv
            self.num_frames_no_success = 0
            self.last_h = Hinv
            img_warped = cv2.warpPerspective(img_query, Hinv, dst_size)
            return True, img_warped, img_flann

    def _match_features(self, desc_frame: np.ndarray) -> List[cv2.DMatch]:
        """Feature matching between train and query image

            This method finds matches between the descriptor of an input
            (query) frame and the stored template (train) image.

            The ratio test is applied to distinguish between good matches and
            outliers.

            :param desc_frame: descriptor of input (query) image
            :returns: list of good matches
        """
        # find 2 best matches (kNN with k=2)
        matches = self.flann.knnMatch(self.desc_train, desc_frame, k=2)
        # discard bad matches, ratio test as per Lowe's paper
        good_matches = [x[0] for x in matches
                        if x[0].distance < 0.7 * x[1].distance]
        return good_matches

    def _detect_corner_points(self,
                              key_frame: np.ndarray,
                              good_matches: Sequence[cv2.DMatch]) -> np.ndarray:
        """Detects corner points in an input (query) image

            This method finds the homography matrix to go from the template
            (train) image to the input (query) image, and finds the coordinates
            of the good matches (from the train image) in the query image.

            :param key_frame: keypoints of the query image
            :param good_matches: list of good matches
            :returns: coordinates of good matches in transformed query image
        """
        # find homography using RANSAC
        src_points = [self.key_train[good_match.queryIdx].pt
                      for good_match in good_matches]
        dst_points = [key_frame[good_match.trainIdx].pt
                      for good_match in good_matches]
        H, _ = cv2.findHomography(np.array(src_points), np.array(dst_points),
                                  cv2.RANSAC)

        if H is None:
            raise Outlier("Homography not found")
        # outline train image in query image
        height, width = self.sh_train
        src_corners = np.array([(0, 0), (width, 0),
                                (width, height),
                                (0, height)], dtype=np.float32)
        return cv2.perspectiveTransform(src_corners[None, :, :], H)[0]

    def _warp_keypoints(self,
                        good_matches: Sequence[cv2.DMatch],
                        key_frame: Sequence[cv2.KeyPoint],
                        sh_frame: Tuple[int,
                                        int]) -> Tuple[np.ndarray,
                                                       Tuple[int,
                                                             int]]:
        """Projects keypoints to the frontal plane

            This method computes the homography matrix that is required to
            project a list of keypoints to the frontal plane.

            :param good_matches: list of good matches
            :param key_frame: list of keypoints in the input (query) image
            :param sh_frame: shape of the input (query) image
            :returns: [Hinv, dst_size] homography matrix and size of resulting
                      image
        """
        # bring object to frontoparallel plane: centered, up-right
        dst_size = (sh_frame[1], sh_frame[0])  # cols,rows
        scale_row = 1. / self.sh_train[0] * dst_size[1] / 2.
        bias_row = dst_size[0] / 4.
        scale_col = 1. / self.sh_train[1] * dst_size[0] * 3 / 4.
        bias_col = dst_size[1] / 8.

        # source points are the ones in the train image
        src_points = [key_frame[good_match.trainIdx].pt
                      for good_match in good_matches]

        # destination points are the ones in the query image
        # off-set in space so that the image is
        dst_points = [self.key_train[good_match.queryIdx].pt
                      for good_match in good_matches]
        dst_points = [[y * scale_row + bias_row, x * scale_col + bias_col]
                      for y, x in dst_points]

        # find homography
        Hinv, _ = cv2.findHomography(np.array(src_points),
                                     np.array(dst_points), cv2.RANSAC)
        return Hinv, dst_size


def draw_good_matches(img1: np.ndarray,
                      kp1: Sequence[cv2.KeyPoint],
                      img2: np.ndarray,
                      kp2: Sequence[cv2.KeyPoint],
                      matches: Sequence[cv2.DMatch]) -> np.ndarray:
    """Visualizes a list of good matches

        This function visualizes a list of good matches. It is only required in
        OpenCV releases that do not ship with the function drawKeypoints.

        The function draws two images (img1 and img2) side-by-side,
        highlighting a list of keypoints in both, and connects matching
        keypoints in the two images with blue lines.

        :param img1: first image
        :param kp1: list of keypoints for first image
        :param img2: second image
        :param kp2: list of keypoints for second image
        :param matches: list of good matches
        :returns: annotated output image
    """
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1, :] = img1[..., None]

    # Place the next image to the right of it
    out[:rows2, cols1:cols1 + cols2, :] = img2[..., None]

    radius = 4
    BLUE = (255, 0, 0)
    thickness = 1

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for m in matches:
        # Get the matching keypoints for each of the images
        # and convert them to int
        c1 = tuple(map(int, kp1[m.queryIdx].pt))
        c2 = tuple(map(int, kp2[m.trainIdx].pt))
        # Shift second center for drawing
        c2 = c2[0] + cols1, c2[1]

        radius = 4
        BLUE = (255, 0, 0)
        thickness = 1
        # Draw a small circle at both co-ordinates
        cv2.circle(out, c1, radius, BLUE, thickness)
        cv2.circle(out, c2, radius, BLUE, thickness)

        # Draw a line in between the two points
        cv2.line(out, c1, c2, BLUE, thickness)

    return out
