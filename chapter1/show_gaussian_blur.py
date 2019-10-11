import cv2
import numpy as np
import matplotlib.pyplot as plt
from tools import dodge_naive
from tools import dodge
from tools import load_img_resized


if __name__ == '__main__':

    dimensions = (512, 512)
    img = load_img_resized('media/lena.png', dimensions)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    small_sample = gray_image[240:245, 240:245]
    print(small_sample.shape)

    fig = plt.figure(figsize=(17, 8))
    fig.add_subplot(1, 2, 1)

    nx, ny = small_sample.shape

    applied = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            x, y = 239 + i, 239 + j
            cut = gray_image[x: x + 3, y: y + 3]
            applied[i, j] = str(round(np.sum(0.1 * cut) + cut[1, 1] * 0.1, 1))

    _filter = [['' for _ in range(nx)] for _ in range(ny)]
    _filter[1][2] = '\n\n0.1'
    _filter[1][3] = '\n\n0.1'
    _filter[1][4] = '\n\n0.1'
    _filter[2][2] = '\n\n0.1'
    _filter[2][3] = '\n\n0.2'
    _filter[2][4] = '\n\n0.1'
    _filter[3][2] = '\n\n0.1'
    _filter[3][3] = '\n\n0.1'
    _filter[3][4] = '\n\n0.1'
    tb = plt.table(cellText=small_sample, loc=(0, 0), cellLoc='center')
    tb2 = plt.table(cellText=_filter, loc=(0, 0), cellLoc='bottom right')

    for cell in tb2._cells:
        tb2._cells[cell].set_alpha(0)

    tc = tb.properties()['child_artists']
    for n, cell in enumerate(tc):
        i, j = n // nx, n % nx
        if 4 > i > 0 and 4 >= j > 1:
            cell.set_color('#f2a391')
        cell.set_height(1 / ny)
        cell.set_width(1 / nx)
        tb2[i, j].set_height(1 / ny)
        tb2[i, j].set_width(1 / nx)

    ax = plt.gca()
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels([0, 1, 2, 3, 4])
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels([4, 3, 2, 1, 0])

    fig.add_subplot(1, 2, 2)
    small_sample = gray_image[240:245, 240:245]
    tb = plt.table(cellText=applied, loc=(0, 0), cellLoc='center')
    tc = tb.properties()['child_artists']
    for n, cell in enumerate(tc):
        cell.set_height(1 / ny)
        cell.set_width(1 / nx)
        i, j = n // nx, n % nx
        if i == 2 and j == 3:
            cell.set_color('#f2a391')

    ax = plt.gca()
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels([0, 1, 2, 3, 4])
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels([4, 3, 2, 1, 0])

    plt.tight_layout()
    plt.show()
