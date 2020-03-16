import argparse
from matplotlib import cm
import itertools
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
from common import load_image

import exifread


MARKERS = ['o', '+', 'x', '*', '.', 'X', '1', 'v', 'D']


def thumbnail(img_rgb, long_edge=400):
    original_long_edge = max(img_rgb.shape[:2])
    dimensions = tuple([int(x / original_long_edge * long_edge) for x in img_rgb.shape[:2]][::-1])
    print('dimensions', dimensions)
    return cv2.resize(img_rgb, dimensions, interpolation=cv2.INTER_AREA)


def exposure_strength(path, iso_ref=100, f_stop_ref=6.375):
    with open(path, 'rb') as infile:
        tags = exifread.process_file(infile)
    [f_stop] = tags['EXIF ApertureValue'].values
    [iso_speed] = tags['EXIF ISOSpeedRatings'].values
    [exposure_time] = tags['EXIF ExposureTime'].values

    rel_aperture_area = 1 / (f_stop.num / f_stop.den / f_stop_ref) ** 2
    exposure_time_float = exposure_time.num / exposure_time.den

    score = rel_aperture_area * exposure_time_float * iso_speed / iso_ref
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
    return img_8bit


OPEN_CV_COLORS = 'bgr'


def plot_crf(crf, colors=OPEN_CV_COLORS):
    for i, c in enumerate(colors):
        plt.plot(crf_debevec[:, 0, i], color=c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    img_group = parser.add_mutually_exclusive_group(required=True)
    img_group.add_argument('--image-dir', type=Path)
    img_group.add_argument('--images', type=Path, nargs='+')
    parser.add_argument('--show-steps', action='store_true')
    parser.add_argument('--random-seed', type=int, default=43)
    parser.add_argument('--num-pixels', type=int, default=100)
    parser.add_argument('--align-images', action='store_true')
    parser.add_argument('--debug-color', choices=OPEN_CV_COLORS, default='g')
    args = parser.parse_args()

    if args.image_dir:
        args.images = sorted(args.image_dir.iterdir())

    args.color_i = OPEN_CV_COLORS.find(args.debug_color)

    images = [load_image(p, bps=8) for p in args.images]
    times = [exposure_strength(p)[0] for p in args.images]
    times_array = np.array(times, dtype=np.float32)
    print('times', times_array)

    if args.show_steps:
        np.random.seed(args.random_seed)
        pixel_values = {}
        while len(pixel_values) < args.num_pixels:
            i = np.random.randint(0, high=images[0].shape[0] - 1)
            j = np.random.randint(0, high=images[0].shape[1] - 1)

            new_val = images[0][i, j, args.color_i]
            good_pixel = True
            for vv in pixel_values.values():
                if np.abs(vv[0].astype(int) - new_val.astype(int)) < 100 // args.num_pixels:
                    good_pixel = False
                    break

            if good_pixel:
                pixel_values[(i, j)] = [img[i, j, args.color_i] for img in images]

        log_ts = [np.log2(t) for t in times]

        for [(i, j), vv], marker in zip(pixel_values.items(), MARKERS):
            plt.scatter(vv, log_ts, marker=marker, label=f'Pixel [{i}, {j}]')
        plt.xlabel('Output Pixel value (8-bit)')
        plt.ylabel('log exposure')
        plt.legend()
        plt.show()

    cal_debevec = cv2.createCalibrateDebevec(samples=200)
    print('Calibrated Debevec')
    crf_debevec = cal_debevec.process(images, times=times_array)

    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(images, times=times_array.copy(), response=crf_debevec)

    print("merged")

    if args.show_steps:
        for [(i, j), vv], marker in zip(pixel_values.items(), MARKERS):
            e = hdr_debevec[i, j, args.color_i]
            plt.scatter(vv, np.array(log_ts) + np.log(e) + 1.6,
                        marker=marker,
                        label=f'Pixel [{i}, {j}]')
        plt.plot(np.log(crf_debevec[:, 0, args.color_i]),
                 color=OPEN_CV_COLORS[args.color_i])
        plt.tight_layout()
        plt.legend()
        plt.show()
    # Tonemap HDR image
    tonemap1 = cv2.createTonemap(gamma=2.2)
    res_debevec = tonemap1.process(hdr_debevec.copy())
    x = save_8bit(res_debevec, 'res_debevec.jpg')
    plt.imshow(x)
    plt.show()

    if args.show_steps:
        merge_robertson = cv2.createMergeRobertson()
        hdr_robertson = merge_robertson.process(images, times=times_array.copy())
        # Tonemap HDR image
        tonemap1 = cv2.createTonemap(gamma=2.2)
        res_robertson = tonemap1.process(hdr_robertson)
        save_8bit(res_robertson, 'res_robertson.jpg')

        # Exposure fusion using Mertens
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(images)
        save_8bit(res_mertens, 'res_mertens.jpg')
