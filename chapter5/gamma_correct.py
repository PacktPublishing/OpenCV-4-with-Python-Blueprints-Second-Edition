import argparse
from pathlib import Path
import itertools
import numpy as np
import matplotlib.pyplot as plt
from common import load_image, load_14bit_gray
import functools


@functools.lru_cache(maxsize=None)
def gamma_transform(x, gamma, bps=14):
    return np.clip(pow(x / 2**bps, gamma) * 255.0, 0, 255)


def apply_gamma(img, gamma, bps=14):
    corrected = img.copy()
    for i, j in itertools.product(range(corrected.shape[0]),
                                  range(corrected.shape[1])):
        corrected[i, j] = gamma_transform(corrected[i, j], gamma, bps=bps)
    return corrected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_image', type=Path,
                        help='Location of a .CR2 file.')
    parser.add_argument('--gamma', type=float, default=0.3)
    args = parser.parse_args()

    gray = load_14bit_gray(args.raw_image)

    normal = np.clip(gray / 64, 0, 255).astype(np.uint8)

    corrected = apply_gamma(gray, args.gamma)

    fig, axes = plt.subplots(2, 2, sharey=False)

    for i, img in enumerate([normal, corrected]):
        axes[1, i].hist(img.flatten(), bins=256)
        axes[1, i].set_ylim(top=1.5e-2 * len(img.flatten()))
        axes[1, i].set_xlabel('Brightness (8 bits)')
        axes[1, i].set_ylabel('Number of pixels')
        axes[0, i].imshow(img, cmap='gray', vmax=255)
    plt.title('Histogram of pixel values')
    plt.savefig('histogram.png')
    plt.show()
