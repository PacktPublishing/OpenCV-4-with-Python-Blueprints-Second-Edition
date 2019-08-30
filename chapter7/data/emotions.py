import csv
import numpy as np
import cv2
import json
import sys
csv.field_size_limit(sys.maxsize)


def encode(label):
    if label == 'neutral':
        return 0
    elif label == 'surprised':
        return 1
    elif label == 'angry':
        return 2
    elif label == 'happy':
        return 3
    elif label == 'sad':
        return 4
    elif label == 'disgusted':
        return 5
    else:
        raise NotImplementedError()


def load_as_training_data(path):
    Y, X = [], []
    with open(path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        for label, sample in reader:
            Y.append(encode(label))
            X.append(np.array(json.loads(sample), dtype=np.float32).flatten())
    return cv2.ml.TrainData_create(np.array(X), cv2.ml.ROW_SAMPLE, np.array(Y))


def save_datum(path, label, img):
    with open(path, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([label, img.tolist()])


if __name__ == '__main__':
    td = load_as_training_data('data/cropped_faces.csv')
    print(td)
