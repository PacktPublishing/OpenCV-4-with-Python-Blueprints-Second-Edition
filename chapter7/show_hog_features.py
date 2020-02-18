import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2


DIMENSIONS = (256, 256)


def _imshow(img):
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='media/00007_00028.ppm')
    args = parser.parse_args()

    img = cv2.resize(cv2.imread(args.image), DIMENSIONS)

    fig, ax = plt.subplots(figsize=[9, 6])

    _imshow(img)

    axins = ax.inset_axes([0.9, 0.5, 0.47, 0.47])
    axins.imshow(img)

    # Indicate where it was zoomed from.
    # x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
    x1, x2, y1, y2 = 128+64, 64+64, 128, 64
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins)

    bins = 9

    # _winSize, Size _blockSize, Size _blockStride, Size _cellSize
    hog = cv2.HOGDescriptor((256, 256), (256, 256), (64, 64), (64, 64), bins)
    features = hog.compute(img)
    print(features.shape)
    x = features.reshape((-1, bins))
    print(np.arange(-180, 180, 40).shape, x[5].shape)
    plt.show()
    plt.bar(np.arange(-180, 180, 40), x[6], width=40)
    plt.xticks(np.arange(-180, 180, 40))
    print(x[5])
    print(x)
    print(x.shape)
    print(x.sum(axis=0))

    plt.show()
