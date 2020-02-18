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

    face_detector = FaceDetector(eye_cascade='../.py3.7-cv-blueprints/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')

    img = cv2.resize(cv2.imread(args.image), DIMENSIONS)

    fig = plt.figure(figsize=(19, 6))
    fig.add_subplot(1, 3, 1)

    success, frame, head, (x, y) = face_detector.detect_face(img)
    assert success, 'Face was not detected'
    _imshow(frame)
    plt.title('Detected Face')

    fig.add_subplot(1, 3, 2)
    head_copy = head.copy()
    eyes = face_detector.eye_centers(head_copy, outline=True)
    _imshow(head_copy)
    plt.title('Detected Eyes')

    fig.add_subplot(1, 3, 3)
    success, trainig_image = face_detector.align_head(head)
    assert success, 'Eyes were not detected.'
    _imshow(trainig_image)
    plt.title('Training Image')

    plt.tight_layout()
    plt.show()
