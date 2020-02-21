import argparse
from matplotlib import cm
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
from .common import load_image

import exifread

'''
https://docs.opencv.org/4.2.0/d2/df0/tutorial_py_hdr.html
'''


def thumbnail(img_rgb, long_edge=400):
    original_long_edge = max(img_rgb.shape[:2])
    dimensions = tuple([int(x / original_long_edge * long_edge) for x in img_rgb.shape[:2]][::-1])
    print('dimensions', dimensions)
    return cv2.resize(img_rgb, dimensions, interpolation=cv2.INTER_AREA)



def exposure_strength(path):
    with open(path, 'rb') as infile:
        tags = exifread.process_file(infile)
    [f_stop] = tags['EXIF ApertureValue'].values
    [iso_speed] = tags['EXIF ISOSpeedRatings'].values
    [exposure_time] = tags['EXIF ExposureTime'].values

    aperture_area = 1 / (f_stop.num / f_stop.den) ** 2
    exposure_time_float = exposure_time.num / exposure_time.den

    score = aperture_area * exposure_time_float * iso_speed
    return score, np.log2(score)


def lowe_match(descriptors1, descriptors2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # discard bad matches, ratio test as per Lowe's paper
    good_matches = [m for m, n in matches
                    if m.distance < 0.7 * n.distance]
    return good_matches


def save_8bit(img, name):
    img_8bit = np.clip(img * 255, 0, 255).astype('uint8')
    cv2.imwrite(name, img_8bit)


def plot_and_show_crf(crf, colors='bgr'):
    for i, c in zip(range(crf.shape[2]), colors):
        plt.plot(crf_debevec[:, 0, i], color=c)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    img_group = parser.add_mutually_exclusive_group(required=True)
    img_group.add_argument('--image-dir', type=Path)
    img_group.add_argument('--images', type=Path, nargs='+')
    parser.add_argument('--show-steps', action='store_true')
    parser.add_argument('--align-images', action='store_true')
    args = parser.parse_args()

    if args.image_dir:
        args.images = sorted(args.image_dir.iterdir())

    images = [load_image(p) for p in args.images]
    times = [exposure_strength(p)[0] for p in args.images]
    times_array = np.array(times, dtype=np.float32)

    cal_debevec = cv2.createCalibrateDebevec()
    crf_debevec = cal_debevec.process(images, times=times_array)
    plot_and_show_crf(crf_debevec)

    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(images, times=times_array.copy())
    # Tonemap HDR image
    tonemap1 = cv2.createTonemap(gamma=2.2)
    res_debevec = tonemap1.process(hdr_debevec.copy())
    save_8bit(res_debevec, 'res_debevec.jpg')

    merge_robertson = cv2.createMergeRobertson()
    hdr_robertson = merge_robertson.process(images, times=times_array.copy())
    print(hdr_robertson)
    # Tonemap HDR image
    tonemap1 = cv2.createTonemap(gamma=2.2)
    res_robertson = tonemap1.process(hdr_robertson)
    save_8bit(res_robertson, 'res_robertson.jpg')

    # Exposure fusion using Mertens
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(images)
    save_8bit(res_mertens, 'res_mertens.jpg')
