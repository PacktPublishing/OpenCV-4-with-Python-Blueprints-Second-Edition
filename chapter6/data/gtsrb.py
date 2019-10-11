#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A module to load the German Traffic Sign Recognition Benchmark (GTSRB)

The dataset contains more than 50,000 images of traffic signs belonging
to more than 40 classes. The dataset can be freely obtained from:
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset.
"""

from pathlib import Path
import requests
from io import TextIOWrapper
import hashlib
import cv2
import numpy as np
from zipfile import ZipFile

import csv
from matplotlib import cm
from matplotlib import pyplot as plt


ARCHIVE_PATH = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/'  # noqa


def _download(filename, *, md5sum=None):
    '''
    GTSRB_Final_Training_Images.zip

    '''
    write_path = Path(__file__).parent / filename
    if write_path.exists() and _md5sum_matches(write_path, md5sum):
        return write_path
    response = requests.get(f'{ARCHIVE_PATH}/{filename}')
    response.raise_for_status()
    with open(write_path, 'wb') as outfile:
        outfile.write(response.content)
    return write_path


def _md5sum_matches(file_path, checksum):
    if checksum is None:
        return True
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return checksum == hash_md5.hexdigest()


def _load_data(filepath, labels):
    data, targets = [], []

    with ZipFile(filepath) as data_zip:
        for path in data_zip.namelist():
            if not path.endswith('.csv'):
                continue
            # Only iterate over annotations files
            *dir_path, csv_filename = path.split('/')
            label_str = dir_path[-1]
            if labels is not None and int(label_str) not in labels:
                continue
            with data_zip.open(path, 'r') as csvfile:
                reader = csv.DictReader(TextIOWrapper(csvfile), delimiter=';')
                for img_info in reader:
                    img_path = '/'.join([*dir_path, img_info['Filename']])
                    raw_data = data_zip.read(img_path)
                    img = cv2.imdecode(np.frombuffer(raw_data, np.uint8), 1)

                    x1, y1 = np.int(img_info['Roi.X1']), np.int(img_info['Roi.Y1'])
                    x2, y2 = np.int(img_info['Roi.X2']), np.int(img_info['Roi.Y2'])

                    data.append(img[y1: y2, x1: x2])
                    targets.append(np.int(img_info['ClassId']))
    return data, targets


def load_test_data(labels=[0, 10]):
    filepath = _download('GTSRB_Online-Test-Images-Sorted.zip',
                         md5sum='b7bba7dad2a4dc4bc54d6ba2716d163b')
    return _load_data(filepath, labels)


def load_training_data(labels=[0, 10]):
    filepath = _download('GTSRB-Training_fixed.zip',
                         md5sum='513f3c79a4c5141765e10e952eaa2478')
    return _load_data(filepath, labels)


if __name__ == '__main__':
    train_data, train_labels = load_training_data(labels=None)
    np.random.seed(75)
    for _ in range(100):
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        for r in range(3):
            for c in range(5):
                i = 5 * r + c
                ax = plt.subplot(3, 5, 1 + i)
                sample = train_data[indices[i]]
                ax.imshow(cv2.resize(sample, (32, 32)), cmap=cm.Greys_r)
                ax.axis('off')
        plt.tight_layout()
        plt.show()
        np.random.seed(np.random.randint(len(indices)))
