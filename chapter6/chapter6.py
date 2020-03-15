#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV with Python Blueprints
Chapter 5: Tracking Visually Salient Objects

An app to track multiple visually salient objects in a video sequence.
"""

import cv2
from os import path

from saliency import get_saliency_map, get_proto_objects_map
from tracking import MultipleObjectsTracker

import time

def main(video_file='soccer.avi', roi=((140, 100), (500, 600))):
    if not path.isfile(video_file):
        print(f'File "{video_file}" does not exist.')
        raise SystemExit

    # open video file
    video = cv2.VideoCapture(video_file)

    # initialize tracker
    mot = MultipleObjectsTracker()

    for _, img in iter(video.read, (False, None)):
        if roi:
            # original video is too big: grab some meaningful ROI
            img = img[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]

        # generate saliency map
        saliency = get_saliency_map(img, use_numpy_fft=False,
                                    gauss_kernel=(3, 3))
        objects = get_proto_objects_map(saliency, use_otsu=False)
        cv2.imshow('original', img)
        cv2.imshow('saliency', saliency)
        cv2.imshow('objects', objects)
        cv2.imshow('tracker', mot.advance_frame(img, objects,saliency))
        # time.sleep(1)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
