import cv2

# ============================================================
# INPUTS
# ============================================================

PAPER_VIDEO = "test_video_tracked_full.mp4"
SORT_VIDEO = "test_video_sort_tracked.mp4"

OUTPUT_VIDEO = "test_video_comparison.avi"


# ============================================================
# OPEN VIDEOS
# ============================================================

paper = cv2.VideoCapture(PAPER_VIDEO)
sort = cv2.VideoCapture(SORT_VIDEO)

if not paper.isOpened():
    raise RuntimeError(f"Could not open {PAPER_VIDEO}")

if not sort.isOpened():
    raise RuntimeError(f"Could not open {SORT_VIDEO}")


# Get properties
fps = paper.get(cv2.CAP_PROP_FPS)

width = int(paper.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(paper.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = min(
    int(paper.get(cv2.CAP_PROP_FRAME_COUNT)),
    int(sort.get(cv2.CAP_PROP_FRAME_COUNT))
)


# ============================================================
# OUTPUT
# ============================================================

combined_width = width * 2

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (combined_width, height)
)


# ============================================================
# PROCESS
# ============================================================

frame_number = 0

while True:

    ret1, frame1 = paper.read()
    ret2, frame2 = sort.read()

    if not ret1 or not ret2:
        break

    frame_number += 1

    # --------------------------------------------------------
    # Add headers
    # --------------------------------------------------------

    cv2.rectangle(
        frame1,
        (0, 0),
        (width, 45),
        (0, 0, 0),
        -1
    )

    cv2.rectangle(
        frame2,
        (0, 0),
        (width, 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame1,
        "PAPER IMPLEMENTATION",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame2,
        "NON-PAPER IMPLEMENTATION",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Frame number
    # --------------------------------------------------------

    cv2.putText(
        frame1,
        f"Frame: {frame_number}",
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame2,
        f"Frame: {frame_number}",
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Combine horizontally
    # --------------------------------------------------------

    combined = cv2.hconcat([
        frame1,
        frame2
    ])

    writer.write(combined)

    # Progress
    if frame_number % 100 == 0:
        print(
            f"Progress: "
            f"{frame_number}/{total_frames}"
        )


# ============================================================
# CLEANUP
# ============================================================

paper.release()
sort.release()
writer.release()

print()
print("=" * 60)
print("COMPARISON VIDEO CREATED")
print("=" * 60)
print(f"Frames: {frame_number}")
print(f"Output: {OUTPUT_VIDEO}")
print("=" * 60)