import argparse
import matplotlib.pyplot as plt
import cv2

from detectors import FaceDetector


DIMENSIONS = (512, 512)


def _imshow(img):
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='media/B04521_07_01.png')
    args = parser.parse_args()

    cascade_clf = cv2.CascadeClassifier('params/haarcascade_frontalface_default.xml')
    img = cv2.resize(cv2.imread(args.image), DIMENSIONS)

    fig = plt.figure(figsize=(13, 6))
    fig.add_subplot(1, 2, 1)
    _imshow(img)
    plt.title('Original Image')

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = cascade_clf.detectMultiScale(gray_img,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_img, (x, y), (x + w, y + h), (100, 255, 0),
                      thickness=2)
    fig.add_subplot(1, 2, 2)
    _imshow(gray_img)
    plt.title('Detected Faces')

    plt.tight_layout()
    plt.show()
