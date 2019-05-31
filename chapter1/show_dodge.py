import cv2
import numpy as np
import matplotlib.pyplot as plt
from tools import dodge_naive
from tools import dodge
from tools import load_img_resized


if __name__ == '__main__':
    dimensions = (512, 512)
    img = load_img_resized('lena.png', dimensions)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(dimensions).astype(np.uint8)
    mask[100:300, 100:300] = 30 * np.ones((200, 200))

    fig = plt.figure(figsize=(17, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
    plt.title('before')
    for i, f in enumerate([dodge], 2):
        fig.add_subplot(1, 2, i)
        plt.imshow(f(gray_image, mask).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
        plt.title(f'after {f.__name__}')
    plt.tight_layout()
    plt.show()
