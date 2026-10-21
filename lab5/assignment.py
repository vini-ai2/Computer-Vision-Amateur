import cv2
from ultralytics import YOLO
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_VIDEO = "road.mp4"
OUTPUT_VIDEO = "road_tracked.mp4"

# Lightweight model - good for real-time / long videos
MODEL = "yolo11n.pt"

# COCO classes that we care about
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

CONFIDENCE = 0.35

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL)

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video information")
print("------------------")
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)

# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# TRACKING DATA
# ============================================================

# Stores trajectory points for each object ID
track_history = defaultdict(list)

# Stores unique IDs by class
unique_ids = defaultdict(set)

# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO DETECTION + BYTETRACK TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            # Class ID
            class_id = int(boxes.cls[i].item())

            # Ignore classes we don't care about
            if class_id not in TARGET_CLASSES:
                continue

            class_name = TARGET_CLASSES[class_id]

            # Confidence
            confidence = float(boxes.conf[i].item())

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                boxes.xyxy[i].tolist()
            )

            # Tracking ID
            if boxes.id is not None:

                track_id = int(boxes.id[i].item())

                unique_ids[class_name].add(track_id)

                # ------------------------------------------------
                # CENTER POINT
                # ------------------------------------------------

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                track_history[track_id].append(
                    (center_x, center_y)
                )

                # Keep only recent trajectory points
                if len(track_history[track_id]) > 50:
                    track_history[track_id].pop(0)

                # ------------------------------------------------
                # DRAW TRAJECTORY
                # ------------------------------------------------

                points = track_history[track_id]

                for j in range(1, len(points)):

                    cv2.line(
                        frame,
                        points[j - 1],
                        points[j],
                        (255, 0, 0),
                        2
                    )

                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

            else:

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # DRAW LABEL BACKGROUND
            # ------------------------------------------------

            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                (0, 255, 0),
                -1
            )

            # ------------------------------------------------
            # DRAW LABEL
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

    # ========================================================
    # DISPLAY FRAME NUMBER
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    out.write(frame)

    # Optional live preview
    cv2.imshow("YOLO + ByteTrack", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()

# ============================================================
# PRINT STATISTICS
# ============================================================

print("\n==============================")
print("TRACKING COMPLETE")
print("==============================")

print("\nUnique objects detected:")

for class_name, ids in unique_ids.items():

    print(
        f"{class_name:12s}: {len(ids)}"
    )

print("\nOutput saved as:")
print(OUTPUT_VIDEO)