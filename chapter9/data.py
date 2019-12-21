import glob
import os

from itertools import count
from collections import defaultdict, namedtuple

import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

DATASET_DIR = "dataset"
for type in ("annotations", "images"):
    tf.keras.utils.get_file(
        type,
        f"https://www.robots.ox.ac.uk/~vgg/data/pets/data/{type}.tar.gz",
        untar=True,
        cache_dir=".",
        cache_subdir=DATASET_DIR)

IMAGE_SIZE = 224
IMAGE_ROOT = os.path.join(DATASET_DIR, "images")
XML_ROOT = os.path.join(DATASET_DIR, "annotations")

Data = namedtuple("Data", "image,box,size,type,breed")

types = defaultdict(count().__next__)
breeds = defaultdict(count().__next__)


def parse_xml(path: str) -> Data:
    with open(path) as f:
        xml_string = f.read()
    root = ET.fromstring(xml_string)
    img_name = root.find("./filename").text
    breed_name = img_name[:img_name.rindex("_")]
    breed_id = breeds[breed_name]
    type_id = types[root.find("./object/name").text]
    box = np.array([int(root.find(f"./object/bndbox/{tag}").text)
                    for tag in "xmin,ymin,xmax,ymax".split(",")])
    size = np.array([int(root.find(f"./size/{tag}").text)
                     for tag in "width,height".split(",")])
    normed_box = (box.reshape((2, 2)) / size).reshape((4))
    return Data(img_name, normed_box, size, type_id, breed_id)


xml_paths = glob.glob(os.path.join(XML_ROOT, "xmls", "*.xml"))
xml_paths.sort()

parsed = np.array([parse_xml(path) for path in xml_paths])

print(f"{len(types)} TYPES:", *types.keys(), sep=", ")
print(f"{len(breeds)} BREEDS:", *breeds.keys(), sep=", ")

np.random.seed(1)
np.random.shuffle(parsed)

ds = tuple(np.array(list(i)) for i in np.transpose(parsed))
ds_slices = tf.data.Dataset.from_tensor_slices(ds)

for el in ds_slices.take(1):
    print(el)
# check boxes
for el in ds_slices:
    b = el[1].numpy()
    if(np.any((b > 1) | (b < 0)) or np.any(b[2:] < b[:2])):
        print(f"Invalid box found {b}")


def prepare(image, box, size, type, breed):
    image = tf.io.read_file(IMAGE_ROOT + "/" + image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image /= 255
    return Data(image, box, size, tf.one_hot(
        type, len(types)), tf.one_hot(breed, len(breeds)))


ds = ds_slices.map(prepare).prefetch(128)

if __name__ == "__main__":
    def illustrate(sample):
        breed_num = np.argmax(sample.breed)
        for breed, num in breeds.items():
            if num == breed_num:
                break
        image = sample.image.numpy()
        pt1, pt2 = (sample.box.numpy().reshape(
            (2, 2)) * IMAGE_SIZE).astype(np.int32)
        cv2.rectangle(image, tuple(pt1), tuple(pt2), (0, 1, 0))
        cv2.putText(image, breed, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 1, 0))
        return image
    samples_image = np.concatenate([illustrate(sample)
                                    for sample in ds.take(3)], axis=1)
    cv2.imshow("samples", samples_image)
    cv2.waitKey(0)
