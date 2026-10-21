import cv2
import time
import torch
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict




INPUT_VIDEO = "test_video.mp4"

# Full-video outputs
OUTPUT_VIDEO = "test_video_tracked_full.mp4"
OUTPUT_CSV = "test_video_tracking_data_full.csv"

MODEL_NAME = "yolo11n.pt"

# None = process the entire video
MAX_FRAMES = None

# Detection confidence
CONFIDENCE = 0.40

TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus", 
    7: "truck"
}




if torch.cuda.is_available():
    DEVICE = "cuda"
    print("GPU detected:")
    print(torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("WARNING: CUDA not available. Using CPU.")




print("\nLoading YOLO model...")

model = YOLO(MODEL_NAME)




cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {INPUT_VIDEO}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)




if MAX_FRAMES is None:
    frames_to_process = total_frames
else:
    frames_to_process = min(MAX_FRAMES, total_frames)


print("\nVideo information")
print("--------------------------")
print(f"FPS              : {fps:.2f}")
print(f"Resolution       : {width} x {height}")
print(f"Total frames     : {total_frames}")
print(f"Frames to process: {frames_to_process}")
print(f"Confidence       : {CONFIDENCE}")
print(f"Device           : {DEVICE}")




fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError(
        "Could not create output video."
    )




track_history = defaultdict(list)

unique_ids = defaultdict(set)

rows = []




start_time = time.time()

frame_number = 0



print("\nStarting detection + tracking...\n")


while frame_number < frames_to_process:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1




    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]



    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            class_id = int(
                boxes.cls[i].item()
            )

            # Ignore irrelevant classes
            if class_id not in TARGET_CLASSES:
                continue

            class_name = TARGET_CLASSES[class_id]

            confidence = float(
                boxes.conf[i].item()
            )

            x1, y1, x2, y2 = map(
                int,
                boxes.xyxy[i].tolist()
            )


            track_id = None

            if boxes.id is not None:

                track_id = int(
                    boxes.id[i].item()
                )

                unique_ids[class_name].add(
                    track_id
                )


                # Center of bounding box
                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )



                track_history[track_id].append(
                    (center_x, center_y)
                )


                # Keep last 50 points
                if len(track_history[track_id]) > 50:

                    track_history[track_id].pop(0)




                points = track_history[track_id]

                for j in range(1, len(points)):

                    cv2.line(
                        frame,
                        points[j - 1],
                        points[j],
                        (255, 0, 0),
                        2
                    )


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




            rows.append({
                "frame": frame_number,
                "class": class_name,
                "track_id": track_id,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })




            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )



            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            label_y = max(
                y1 - 5,
                text_h + 5
            )


            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y - text_h - 5
                ),
                (
                    x1 + text_w,
                    label_y
                ),
                (0, 255, 0),
                -1
            )




            cv2.putText(
                frame,
                label,
                (x1, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )



    elapsed = time.time() - start_time

    processing_fps = (
        frame_number / elapsed
        if elapsed > 0
        else 0
    )


    # Display progress on video
    cv2.putText(
        frame,
        f"Frame: {frame_number}/{frames_to_process}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Processing FPS: {processing_fps:.1f}",
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(frame)


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 100 == 0:

        progress = (
            frame_number / frames_to_process
        ) * 100

        print(
            f"Progress: "
            f"{progress:5.1f}% | "
            f"Frame {frame_number}/{frames_to_process} | "
            f"FPS: {processing_fps:.1f}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

total_time = time.time() - start_time

average_fps = (
    frame_number / total_time
    if total_time > 0
    else 0
)


print("\n")
print("=" * 50)
print("FULL VIDEO TRACKING COMPLETE")
print("=" * 50)

print(f"\nFrames processed : {frame_number}")
print(f"Processing time  : {total_time:.2f} sec")
print(f"Average FPS      : {average_fps:.2f}")

print("\nUnique tracked objects:")

for class_name in TARGET_CLASSES.values():

    count = len(
        unique_ids[class_name]
    )

    if count > 0:

        print(
            f"  {class_name:12s}: {count}"
        )


print("\nFiles created:")
print(f"  {OUTPUT_VIDEO}")
print(f"  {OUTPUT_CSV}")

print("\nDone!")