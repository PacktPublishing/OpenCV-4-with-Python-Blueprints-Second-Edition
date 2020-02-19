import argparse
import time

import cv2
import numpy as np

# Define Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (20, 200, 20)
RED = (20, 20, 255)

# Define trackers
trackers = {
    'BOOSTING': cv2.TrackerBoosting_create,
    'MIL': cv2.TrackerMIL_create,
    'KCF': cv2.TrackerKCF_create,
    'TLD': cv2.TrackerTLD_create,
    'MEDIANFLOW': cv2.TrackerMedianFlow_create,
    'GOTURN': cv2.TrackerGOTURN_create,
    'MOSSE': cv2.TrackerMOSSE_create,
    'CSRT': cv2.TrackerCSRT_create

}

# Parse arguments
parser = argparse.ArgumentParser(description='Tracking API demo.')
parser.add_argument(
    '--tracker',
    default="KCF",
    help=f"One of {trackers.keys()}")
parser.add_argument(
    '--video',
    help="Video file to use",
    default="videos/test.mp4")
args = parser.parse_args()


tracker_name = args.tracker.upper()
assert tracker_name in trackers, f"Tracker should be one of {trackers.keys()}"
# Open the video and read the first frame
video = cv2.VideoCapture(args.video)
assert video.isOpened(), "Could not open video"
ok, frame = video.read()
assert ok, "Video file is not readable"

# Select bounding box
bbox = cv2.selectROI(frame, False)

# Initialize the tracker
tracker = trackers[tracker_name]()
tracker.init(frame, bbox)

for ok, frame in iter(video.read, (False, None)):
    # Time in seconds
    start_time = time.time()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate FPS
    fps = 1 / (time.time() - start_time)

    # Display tracking info and show frame
    if ok:
        # Draw bounding box
        x, y, w, h = np.array(bbox, dtype=np.int)
        cv2.rectangle(frame, (x, y), (x + w, y + w), GREEN, 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failed", (100, 80), FONT, 0.7, RED, 2)
    cv2.putText(frame, f"{tracker_name} Tracker",
                (100, 20), FONT, 0.7, GREEN, 2)
    cv2.putText(frame, f"FPS : {fps:.0f}", (100, 50), FONT, 0.7, GREEN, 2)
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xff == 27:
        break
