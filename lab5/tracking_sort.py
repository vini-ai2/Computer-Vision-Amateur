import cv2
import csv
import time
import numpy as np

from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment



INPUT_VIDEO = "test_video.mp4"
OUTPUT_VIDEO = "test_video_sort_tracked.mp4"
OUTPUT_CSV = "test_video_tracking_data_sort.csv"

MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.45

TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus"
}

MAX_AGE = 30
MIN_HITS = 3
IOU_THRESHOLD = 0.3

MAX_FRAMES = None


#kalman filter

class Track:
    next_id = 0

    def __init__(self, bbox, class_id):
        self.bbox = np.array(bbox, dtype=float)
        self.class_id = class_id

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State:
        # [x, y, s, r, vx, vy, vs]
        #
        # x = center x
        # y = center y
        # s = area
        # r = aspect ratio

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=float)

        self.kf.R *= 10
        self.kf.P *= 10
        self.kf.Q *= 0.01

        self.kf.x[:4] = self.bbox_to_state(bbox).reshape(4, 1)

        self.id = Track.next_id
        Track.next_id += 1

        self.age = 0
        self.hits = 1
        self.time_since_update = 0



    @staticmethod
    def bbox_to_state(bbox):

        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1

        x = x1 + w / 2
        y = y1 + h / 2

        area = w * h
        aspect_ratio = w / float(h + 1e-6)

        return np.array([
            x,
            y,
            area,
            aspect_ratio
        ])


    def state_to_bbox(self):

        x, y, area, ratio = self.kf.x[:4].flatten()

        area = max(area, 1)
        ratio = max(ratio, 0.1)

        w = np.sqrt(area * ratio)
        h = area / w

        return np.array([
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2
        ])


    def predict(self):

        self.kf.predict()

        self.age += 1
        self.time_since_update += 1

        self.bbox = self.state_to_bbox()

        return self.bbox


    def update(self, bbox):

        measurement = self.bbox_to_state(bbox)

        self.kf.update(measurement.reshape(4, 1))

        self.bbox = np.array(bbox)

        self.time_since_update = 0
        self.hits += 1


#iou

def calculate_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area_a = (
        (box_a[2] - box_a[0]) *
        (box_a[3] - box_a[1])
    )

    area_b = (
        (box_b[2] - box_b[0]) *
        (box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0

    return intersection / union




def associate_detections_to_tracks(
        detections,
        tracks,
        iou_threshold
):

    if len(tracks) == 0:
        return [], list(range(len(detections))), []

    if len(detections) == 0:
        return [], [], list(range(len(tracks)))

    iou_matrix = np.zeros(
        (len(detections), len(tracks)),
        dtype=float
    )

    for d, detection in enumerate(detections):

        for t, track in enumerate(tracks):

            # Don't match different object classes
            if detection["class_id"] != track.class_id:
                continue

            iou_matrix[d, t] = calculate_iou(
                detection["bbox"],
                track.bbox
            )

    # Hungarian algorithm minimizes cost.
    # Convert IoU into cost.
    cost_matrix = 1 - iou_matrix

    detection_indices, track_indices = linear_sum_assignment(
        cost_matrix
    )

    matches = []

    unmatched_detections = set(
        range(len(detections))
    )

    unmatched_tracks = set(
        range(len(tracks))
    )

    for d, t in zip(
        detection_indices,
        track_indices
    ):

        if iou_matrix[d, t] < iou_threshold:
            continue

        matches.append((d, t))

        unmatched_detections.discard(d)
        unmatched_tracks.discard(t)

    return (
        matches,
        list(unmatched_detections),
        list(unmatched_tracks)
    )




print("=" * 60)
print("YOLO + CUSTOM SORT TRACKER")
print("=" * 60)

model = YOLO(MODEL_NAME)

device = 0 if __import__("torch").cuda.is_available() else "cpu"

print(f"Device: {device}")

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open {INPUT_VIDEO}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

frames_to_process = (
    total_frames
    if MAX_FRAMES is None
    else min(MAX_FRAMES, total_frames)
)

print(f"Video FPS: {fps:.2f}")
print(f"Resolution: {width} x {height}")
print(f"Total frames: {total_frames}")
print(f"Processing: {frames_to_process} frames")


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)



csv_file = open(
    OUTPUT_CSV,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "class",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
])




tracks = []

unique_ids = {
    class_name: set()
    for class_name in TARGET_CLASSES.values()
}

track_detection_count = {}

frame_number = 0

start_time = time.time()




while frame_number < frames_to_process:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

 

    for track in tracks:
        track.predict()



    results = model(
        frame,
        conf=CONFIDENCE,
        device=device,
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for i in range(len(boxes)):

            class_id = int(
                boxes.cls[i].item()
            )

            confidence = float(
                boxes.conf[i].item()
            )

            if class_id not in TARGET_CLASSES:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy()

            detections.append({
                "bbox": xyxy,
                "class_id": class_id,
                "confidence": confidence
            })



    matches, unmatched_detections, unmatched_tracks = (
        associate_detections_to_tracks(
            detections,
            tracks,
            IOU_THRESHOLD
        )
    )



    for detection_index, track_index in matches:

        detection = detections[detection_index]

        track = tracks[track_index]

        track.update(
            detection["bbox"]
        )



    for detection_index in unmatched_detections:

        detection = detections[detection_index]

        new_track = Track(
            detection["bbox"],
            detection["class_id"]
        )

        tracks.append(new_track)


    tracks = [
        track
        for track in tracks
        if track.time_since_update <= MAX_AGE
    ]

#drawing the identified tracks

    for track in tracks:

        if track.time_since_update > 0:
            continue

        if track.hits < MIN_HITS:
            continue

        x1, y1, x2, y2 = map(
            int,
            track.bbox
        )

        class_name = TARGET_CLASSES[
            track.class_id
        ]

        track_id = track.id

        unique_ids[class_name].add(
            track_id
        )

        key = (class_name, track_id)

        track_detection_count[key] = (
            track_detection_count.get(key, 0) + 1
        )

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = (
            f"{class_name} "
            f"ID:{track_id}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # CSV
        confidence = 0.0

        csv_writer.writerow([
            frame_number,
            class_name,
            track_id,
            confidence,
            x1,
            y1,
            x2,
            y2
        ])



    if frame_number % 100 == 0:

        progress = (
            frame_number /
            frames_to_process
        ) * 100

        print(
            f"Progress: "
            f"{progress:.1f}% "
            f"({frame_number}/"
            f"{frames_to_process})"
        )

    writer.write(frame)




cap.release()
writer.release()
csv_file.close()

elapsed = time.time() - start_time

processing_fps = (
    frames_to_process / elapsed
    if elapsed > 0
    else 0
)




print()
print("=" * 60)
print("SORT TRACKING COMPLETE")
print("=" * 60)

print(f"Frames processed : {frame_number}")
print(f"Processing time  : {elapsed:.2f} sec")
print(f"Average FPS      : {processing_fps:.2f}")

print()
print("Unique SORT track IDs:")

for class_name in TARGET_CLASSES.values():

    count = len(
        unique_ids[class_name]
    )

    if count > 0:
        print(
            f"  {class_name:12s}: {count}"
        )

print()
print("Files created:")
print(f"  {OUTPUT_VIDEO}")
print(f"  {OUTPUT_CSV}")

print("=" * 60)