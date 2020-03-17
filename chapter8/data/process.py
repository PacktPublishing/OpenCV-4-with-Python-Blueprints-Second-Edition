import json
import numpy as np
from typing import Callable
import cv2


def featurize(datum):
    return np.array(datum, dtype=np.float32).flatten()


EMOTIONS = {
    'neutral': 0,
    'surprised': 1,
    'angry': 2,
    'happy': 3,
    'sad': 4,
    'disgusted': 5
}

REVERSE_EMOTIONS = {v: k for k, v in EMOTIONS.items()}


def int_encode(label):
    return EMOTIONS[label]


def int_decode(value):
    return REVERSE_EMOTIONS[value]


def one_hot_encode(all_labels) -> (np.ndarray, Callable):
    unique_lebels = list(sorted(set(all_labels)))
    index_to_label = dict(enumerate(unique_lebels))
    label_to_index = {v: k for k, v in index_to_label.items()}

    y = np.zeros((len(all_labels), len(unique_lebels))).astype(np.float32)
    for i, label in enumerate(all_labels):
        y[i, label_to_index[label]] = 1

    return y, index_to_label


def train_test_split(n, train_portion=0.8, seed=None):
    if seed:
        np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)
    N = int(n * train_portion)
    return indices[:N], indices[N:]


def _pca_featurize(data, center, top_vecs):
    return np.array([np.dot(top_vecs, np.array(datum).flatten() - center)
                     for datum in data]).astype(np.float32)


def pca_featurize(training_data, *, num_components=20):
    x_arr = np.array(training_data).reshape((len(training_data), -1)).astype(np.float32)
    mean, eigvecs = cv2.PCACompute(x_arr, mean=None)

    # Take only first num_components eigenvectors.
    top_vecs = eigvecs[:num_components]
    center = mean.flatten()

    args = (center, top_vecs)
    return _pca_featurize(training_data, *args), args


if __name__ == '__main__':
    print(train_test_split(10, 0.8))
    from data.store import load_collected_data
    data, targets = load_collected_data('data/cropped_faces.csv')
    X, f = pca_featurize(data)
    print(X.shape)

