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
    # Dictionaries form names to either data or targets.
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


def download_IJCNN_sorted_test_data():
    return _download('GTSRB_Online-Test-Images-Sorted.zip')


def load_test_data(labels=[0, 10]):
    filepath = download_IJCNN_sorted_test_data()
    return _load_data(filepath, labels)


def download_IJCNN_training_data():
    return _download('GTSRB-Training_fixed.zip',
                     md5sum='513f3c79a4c5141765e10e952eaa2478')


def load_training_data(labels=[0, 10]):
    filepath = download_IJCNN_training_data()
    return _load_data(filepath, labels)


if __name__ == '__main__':
    download_IJCNN_training_data()
    download_IJCNN_sorted_test_data()
    # X, Y = load_training_data()
