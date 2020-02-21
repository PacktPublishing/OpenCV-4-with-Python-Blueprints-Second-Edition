import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from common import load_image, load_14bit_gray


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, nargs=2,
                        help='Location of a .CR2 file.')
    args = parser.parse_args()

    images = [load_14bit_gray(p) for p in args.images]
    fig, axes = plt.subplots(2, len(images), sharey=False)
    for i, gray in enumerate(images):
        axes[1, i].hist(gray.flatten(), bins=256)
        axes[1, i].set_ylim(top=1.5e-2 * len(gray.flatten()))
        axes[1, i].set_xlabel('Brightness (14 bits)')
        axes[1, i].set_ylabel('Number of pixels')
        axes[0, i].imshow(gray, cmap='gray', vmax=2**14)
    plt.title('Histogram of pixel values')
    plt.savefig('histogram.png')
    plt.show()
