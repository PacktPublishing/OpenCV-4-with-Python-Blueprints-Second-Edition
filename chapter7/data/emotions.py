import csv
import numpy as np
import cv2
import json
from enum import IntEnum, auto, unique
import sys
csv.field_size_limit(sys.maxsize)


@unique
class Emotion(IntEnum):
    neutral = auto()


EMOTIONS = {
    'neutral': 0,
    'surprised': 1,
    'angry': 2,
    'happy': 3,
    'sad': 4,
    'disgusted': 5
}

REVERSE_EMOTIONS = {v: k for k, v in EMOTIONS.items()}


def encode(label):
    return EMOTIONS[label]


def decode(value):
    return REVERSE_EMOTIONS[value]


def featurize(datum):
    return np.array(datum, dtype=np.float32).flatten()


def load_as_training_data(path):
    Y, X = [], []
    with open(path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        for label, sample in reader:
            Y.append(encode(label))
            X.append(featurize(json.loads(sample)))
    return cv2.ml.TrainData_create(np.array(X), cv2.ml.ROW_SAMPLE, np.array(Y))


def save_datum(path, label, img):
    with open(path, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([label, img.tolist()])


if __name__ == '__main__':
    td = load_as_training_data('data/cropped_faces.csv')
    print(td)
