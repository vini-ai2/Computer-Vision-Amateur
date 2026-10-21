
import cv2
import numpy as np
from pathlib import Path

# ---------- CONFIGURATION ----------
IMAGE_DIR = Path("calibration_images")

# Set this to the number of INNER CORNERS (not squares).
# Example: 9 columns x 6 rows of inner corners.
CHECKERBOARD = (6, 4)

# Your measured full square side: 5 cm = 50 mm
SQUARE_SIZE_MM = 50.0

# Output files
OUTPUT_FILE = "camera_calibration.npz"
DETECTIONS_DIR = Path("calibration_debug")


def main():
    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in {IMAGE_DIR.resolve()}"
        )

    # Prepare checkerboard 3D coordinates.
    # The Z coordinate is zero because the board is planar.
    objp = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
        dtype=np.float32
    )

    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE_MM

    object_points = []
    image_points = []
    image_size = None

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"[SKIP] Cannot read: {path.name}")
            failed.append(path.name)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]
        elif gray.shape[::-1] != image_size:
            print(f"[SKIP] Different resolution: {path.name}")
            failed.append(path.name)
            continue

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print(f"[FAIL] Corners not detected: {path.name}")
            failed.append(path.name)
            continue

        # Refine corner locations to subpixel accuracy.
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        object_points.append(objp.copy())
        image_points.append(corners)
        successful.append(path.name)

        # Save an image showing detected corners.
        debug_image = image.copy()
        cv2.drawChessboardCorners(
            debug_image, CHECKERBOARD, corners, found
        )
        cv2.imwrite(
            str(DETECTIONS_DIR / path.name),
            debug_image
        )

        print(f"[OK] {path.name}")

    print(f"\nDetected: {len(successful)}/{len(image_paths)}")

    # Require enough successful views for a useful calibration.
    if len(successful) < 10:
        raise RuntimeError(
            "Fewer than 10 usable images. Check CHECKERBOARD "
            "and capture more sharp, varied views."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )
    )


    # Calculate per-view reprojection errors.
    per_view_errors = []

    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        observed = image_points[i].reshape(-1, 2).astype(np.float64)
        predicted = projected.reshape(-1, 2).astype(np.float64)

        # RMS Euclidean error per detected corner, in pixels
        error = np.sqrt(
            np.mean(np.sum((observed - predicted) ** 2, axis=1))
        )

        per_view_errors.append(error)


    print("\n========== CALIBRATION RESULTS ==========")
    print("Image size (width, height):", image_size)
    print("Successful images:", len(successful))
    print("\nCamera matrix K:\n", camera_matrix)
    print("\nDistortion coefficients:\n", dist_coeffs.ravel())
    print(f"\nOverall RMS reprojection error: {rms:.4f} px")
    print(
        "Mean per-view reprojection error: "
        f"{np.mean(per_view_errors):.4f} px"
    )

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        square_size_mm=SQUARE_SIZE_MM,
        checkerboard=np.array(CHECKERBOARD),
        rms_error=rms
    )

    print(f"\nSaved calibration to: {OUTPUT_FILE}")
    print(f"Corner visualizations saved in: {DETECTIONS_DIR.resolve()}")
    print("\nFailed images:", failed)


if __name__ == "__main__":
    main()
