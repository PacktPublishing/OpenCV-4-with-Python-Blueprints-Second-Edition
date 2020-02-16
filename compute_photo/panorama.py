import argparse
from matplotlib import cm
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

import rawpy
from hdr import load_image


def thumbnail(img_rgb, long_edge=400):
    original_long_edge = max(img_rgb.shape[:2])
    dimensions = tuple([int(x / original_long_edge * long_edge) for x in img_rgb.shape[:2]][::-1])
    print('dimensions', dimensions)
    return cv2.resize(img_rgb, dimensions, interpolation=cv2.INTER_AREA)


def read_cr2(path):
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(no_auto_bright=True, output_bps=8)


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


def find_homography(img_gray, kp_base, desc_base):
    kp, desc = f_extractor.detectAndCompute(img_gray, None)

    good_matches = lowe_match(desc, desc_base)

    base_pts = np.float32([kp_base[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(pts, base_pts, cv2.RANSAC, 5.0)
    return M


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    img_group = parser.add_mutually_exclusive_group(required=True)
    img_group.add_argument('--image-dir', type=Path)
    img_group.add_argument('--images', type=Path, nargs='+')
    parser.add_argument('--show-steps', action='store_true')
    args = parser.parse_args()

    if args.image_dir:
        args.images = sorted(args.image_dir.iterdir())

    images = [load_image(p, bps=8) for p in args.images]

    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    print('status', status)
    # cv2.imshow('Panorama', stitched)
    plt.imshow(stitched)
    plt.show()
