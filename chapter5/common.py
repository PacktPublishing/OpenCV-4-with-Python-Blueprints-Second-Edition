import rawpy
import cv2
import numpy as np


def load_image(path, bps=16):
    if path.suffix == '.CR2':
        with rawpy.imread(str(path)) as raw:
            data = raw.postprocess(no_auto_bright=True,
                                   gamma=(1, 1),
                                   output_bps=bps)
        return data
    else:
        return cv2.imread(str(path))


def load_14bit_gray(path):
    img = load_image(path, bps=16)
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 4).astype(np.uint16)
