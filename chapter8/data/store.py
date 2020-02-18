import csv
import pickle
import json
from enum import IntEnum, auto, unique
import sys
csv.field_size_limit(sys.maxsize)


def load_collected_data(path):
    data, targets = [], []
    with open(path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        for label, sample in reader:
            targets.append(label)
            data.append(json.loads(sample))
    return data, targets


def save_datum(path, label, img):
    with open(path, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([label, img.tolist()])


def pickle_dump(f, path):
    with open(path, 'wb') as outfile:
        return pickle.dump(f, outfile)


def pickle_load(path):
    with open(path, 'rb') as infile:
        return pickle.load(infile)


if __name__ == '__main__':
    td = load_collected_data('data/cropped_faces.csv')
    print([len(x) for x in td])
