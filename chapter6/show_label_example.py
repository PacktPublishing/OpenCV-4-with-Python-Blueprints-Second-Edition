import argparse
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2

from data.gtsrb import load_training_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=int, default=30)
    args = parser.parse_args()

    train_data, train_labels = load_training_data(labels=None)
    for datum, label in zip(train_data, train_labels):
        if label == args.label:
            plt.imshow(datum, cmap=cm.Greys_r)
            plt.show()
