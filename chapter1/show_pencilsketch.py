import argparse
import cv2
import matplotlib.pyplot as plt
from tools import pencil_sketch_on_canvas
from tools import load_img_resized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--standalone', action='store_true')
    parser.add_argument('--canvas')
    parser.add_argument('--source', default='lena.png')
    args = parser.parse_args()

    dimensions = (512, 512)
    img = load_img_resized(args.source, dimensions)
    if args.canvas is not None:
        color_canvas = load_img_resized(args.canvas, dimensions)
        gray_canvas = cv2.cvtColor(color_canvas, cv2.COLOR_RGB2GRAY)
    else:
        gray_canvas = None

    if args.standalone:
        fig = plt.figure(figsize=(8, 8))
    else:
        fig = plt.figure(figsize=(17, 8))
        fig.add_subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
        plt.title('before')
        fig.add_subplot(1, 2, 2)
    plt.imshow(pencil_sketch_on_canvas(img, canvas=gray_canvas), vmin=0, vmax=255)
    plt.title('after convert_to_pencil_sketch' + (' with canvas' if args.canvas else ''))
    plt.tight_layout()
    plt.show()
