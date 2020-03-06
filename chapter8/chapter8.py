#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
OpenCV with Python Blueprints
Chapter 8: Learning to Recognize Emotion in Faces

    An app that combines both face detection and face recognition, with a
    focus on recognizing emotional expressions in the detected faces.

    The process flow is as follows:
    * Run the GUI in Training Mode to assemble a training set. Upon exiting
      the app will dump all assembled training samples to a pickle file
      "datasets/faces_training.pkl".
    * Run the script train_test_mlp.py to train a MLP classifier on the
      dataset. This file will store the parameters of the trained MLP in
      a file "params/mlp.xml" and dump the preprocessed dataset to a
      pickle file "datasets/faces_preprocessed.pkl".
    * Run the GUI in Testing Mode to apply the pre-trained MLP classifier
      to the live stream of the webcam.
"""

import argparse
import cv2
import numpy as np

import wx
from pathlib import Path

from data.store import save_datum, pickle_load
from data.process import _pca_featurize
from detectors import FaceDetector
from wx_gui import BaseLayout


class FacialExpressionRecognizerLayout(BaseLayout):
    def __init__(self, *args,
                 clf_path=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.clf = cv2.ml.ANN_MLP_load(str(clf_path / 'mlp.xml'))

        self.index_to_label = pickle_load(clf_path / 'index_to_label')
        self.pca_args = pickle_load(clf_path / 'pca_args')

        self.face_detector = FaceDetector(
            face_cascade='params/haarcascade_frontalface_default.xml',
            eye_cascade='params/haarcascade_lefteye_2splits.xml')

    def featurize_head(self, head):
        return _pca_featurize(head[None], *self.pca_args)

    def augment_layout(self):
        pass

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        success, frame, self.head, (x, y) = self.face_detector.detect_face(
            frame_rgb)
        if not success:
            return frame

        success, head = self.face_detector.align_head(self.head)
        if not success:
            return frame

        # We have to pass [1 x n] array predict.
        _, output = self.clf.predict(self.featurize_head(head))
        label = self.index_to_label[np.argmax(output)]

        # Draw predicted label above the bounding box.
        cv2.putText(frame, label, (x, y - 20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return frame


class DataCollectorLayout(BaseLayout):

    def __init__(self, *args,
                 training_data='data/cropped_faces.csv',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.face_detector = FaceDetector(
            face_cascade='params/haarcascade_frontalface_default.xml',
            eye_cascade='params/haarcascade_lefteye_2splits.xml')

        self.training_data = training_data

    def augment_layout(self):
        """Initializes GUI"""
        # initialize data structure
        self.samples = []
        self.labels = []

        # create a horizontal layout with all buttons
        pnl2 = wx.Panel(self, -1)
        self.neutral = wx.RadioButton(pnl2, -1, 'neutral', (10, 10),
                                      style=wx.RB_GROUP)
        self.happy = wx.RadioButton(pnl2, -1, 'happy')
        self.sad = wx.RadioButton(pnl2, -1, 'sad')
        self.surprised = wx.RadioButton(pnl2, -1, 'surprised')
        self.angry = wx.RadioButton(pnl2, -1, 'angry')
        self.disgusted = wx.RadioButton(pnl2, -1, 'disgusted')
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.neutral, 1)
        hbox2.Add(self.happy, 1)
        hbox2.Add(self.sad, 1)
        hbox2.Add(self.surprised, 1)
        hbox2.Add(self.angry, 1)
        hbox2.Add(self.disgusted, 1)
        pnl2.SetSizer(hbox2)

        # create horizontal layout with single snapshot button
        pnl3 = wx.Panel(self, -1)
        self.snapshot = wx.Button(pnl3, -1, 'Take Snapshot')
        self.Bind(wx.EVT_BUTTON, self._on_snapshot, self.snapshot)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(self.snapshot, 1)
        pnl3.SetSizer(hbox3)

        # arrange all horizontal layouts vertically
        self.panels_vertical.Add(pnl2, flag=wx.EXPAND | wx.BOTTOM, border=1)
        self.panels_vertical.Add(pnl3, flag=wx.EXPAND | wx.BOTTOM, border=1)

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Add a bounding box around the face if a face is detected.
        """
        _, frame, self.head, _ = self.face_detector.detect_face(frame_rgb)
        return frame

    def _on_snapshot(self, evt):
        """Takes a snapshot of the current frame

            This method takes a snapshot of the current frame, preprocesses
            it to extract the head region, and upon success adds the data
            sample to the training set.
        """
        if self.neutral.GetValue():
            label = 'neutral'
        elif self.happy.GetValue():
            label = 'happy'
        elif self.sad.GetValue():
            label = 'sad'
        elif self.surprised.GetValue():
            label = 'surprised'
        elif self.angry.GetValue():
            label = 'angry'
        elif self.disgusted.GetValue():
            label = 'disgusted'

        if self.head is None:
            print("No face detected")
        else:
            success, aligned_head = self.face_detector.align_head(self.head)
            if success:
                save_datum(self.training_data, label, aligned_head)
                print(f"Saved {label} training datum.")
            else:
                print("Could not align head (eye detection failed?)")


def run_layout(layout_cls, **kwargs):
    # open webcam
    capture = cv2.VideoCapture(0)
    # opening the channel ourselves, if it failed to open.
    if not(capture.isOpened()):
        capture.open()

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = layout_cls(capture, **kwargs)
    layout.Center()
    layout.Show()
    app.MainLoop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['collect', 'demo'])
    parser.add_argument('--classifier', type=Path)
    args = parser.parse_args()

    if args.mode == 'collect':
        run_layout(DataCollectorLayout, title='Collect Data')
    elif args.mode == 'demo':
        assert args.classifier is not None, 'you have to provide --classifier'
        run_layout(FacialExpressionRecognizerLayout,
                   title='Facial Expression Recognizer',
                   clf_path=args.classifier)
