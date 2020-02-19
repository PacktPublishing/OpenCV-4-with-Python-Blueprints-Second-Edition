import argparse

import cv2
import numpy as np

from classes import CLASSES_90
from sort import Sort


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    help="Video path, stream URI, or camera ID ", default="demo.mp4")
parser.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="Minimum score to consider")
parser.add_argument("-m", "--mode", choices=['detection', 'tracking'], default="tracking",
                    help="Either detection or tracking mode")

args = parser.parse_args()

if args.input.isdigit():
    args.input = int(args.input)


TRACKED_CLASSES = ["car", "person"]
BOX_COLOR = (23, 230, 210)
TEXT_COLOR = (255, 255, 255)
INPUT_SIZE = (300, 300)

# Read SSD model
config = "./ssd_mobilenet_v1_coco_2017_11_17.pbtxt.txt"
model = "frozen_inference_graph.pb"
detector = cv2.dnn.readNetFromTensorflow(model, config)


def illustrate_box(image: np.ndarray, box: np.ndarray, caption: str) -> None:
    rows, cols = frame.shape[:2]
    points = box.reshape((2, 2)) * np.array([cols, rows])
    p1, p2 = points.astype(np.int32)
    cv2.rectangle(image, tuple(p1), tuple(p2), BOX_COLOR, thickness=4)
    cv2.putText(
        image,
        caption,
        tuple(p1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        TEXT_COLOR,
        2)


def illustrate_detections(dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
    class_ids, scores, boxes = dets[:, 0], dets[:, 1], dets[:, 2:6]
    for class_id, score, box in zip(class_ids, scores, boxes):
        illustrate_box(frame, box, f"{CLASSES_90[int(class_id)]} {score:.2f}")
    return frame


def illustrate_tracking_info(frame: np.ndarray) -> np.ndarray:
    for num, (class_id, tracker) in enumerate(trackers.items()):
        txt = f"{CLASSES_90[class_id]}:Total:{tracker.count} Now:{len(tracker.trackers)}"
        cv2.putText(frame, txt, (0, 50 * (num + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_COLOR, 2)
    return frame


trackers = {CLASSES_90.index(tracked_class): Sort()
            for tracked_class in TRACKED_CLASSES}


def track(dets: np.ndarray,
          illustration_frame: np.ndarray = None) -> np.ndarray:
    for class_id, tracker in trackers.items():
        class_dets = dets[dets[:, 0] == class_id]
        # Retuns [box..,id]
        sort_boxes = tracker.update(class_dets[:, 2:6])
        if illustration_frame is not None:
            for box in sort_boxes:
                illustrate_box(illustration_frame,
                               box[:4],
                               f"{CLASSES_90[class_id]} {int(box[4])}")

    return illustration_frame


cap = cv2.VideoCapture(args.input)

for res, frame in iter(cap.read, (False, None)):
    detector.setInput(
        cv2.dnn.blobFromImage(
            frame,
            size=INPUT_SIZE,
            swapRB=True,
            crop=False))
    detections = detector.forward()[0, 0, :, 1:]
    scores = detections[:, 1]
    detections = detections[scores > 0.3]
    if args.mode == "detection":
        out = illustrate_detections(detections, frame)
    else:
        out = track(detections, frame)
        illustrate_tracking_info(out)
    cv2.imshow("out", out)
    if cv2.waitKey(1) == 27:
        cv2.waitKey(0)
        # exit()
