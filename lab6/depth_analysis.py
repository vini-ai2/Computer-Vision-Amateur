
import cv2
import numpy as np
from pathlib import Path
from itertools import combinations

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("bottle_images")
CALIBRATION_FILE = Path("camera_calibration.npz")

IMAGE_FILES = ["pose1.jpeg", "pose2.jpeg", "pose3.jpeg"]

# Replace with your actual measured camera positions (cm).
CAMERA_X_CM = [0.0, 50.0, 100.0]

# Replace with your actual/estimated camera yaw angles (degrees).
CAMERA_YAW_DEG = [31.0, 0.0, -31.0]

# Replace with your measured ground-truth perpendicular depth.
GROUND_TRUTH_DEPTH_CM = 68.5

# ============================================================


def load_calibration():
    data = np.load(CALIBRATION_FILE)

    K = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64)
    calibration_size = tuple(int(v) for v in data["image_size"])

    return K, dist, calibration_size


def rotation_yaw(degrees):
    """Camera-to-world rotation for yaw around world Y."""
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float64)


def camera_pose(x_cm, yaw_deg):
    """Return camera centre, world-to-camera rotation, translation and P."""
    C = np.array([x_cm, 0.0, 0.0], dtype=np.float64)

    R_cw = rotation_yaw(yaw_deg)
    R_wc = R_cw.T
    t_wc = -R_wc @ C

    P = np.hstack((R_wc, t_wc.reshape(3, 1)))
    return C, R_wc, t_wc, P


def click_feature(image, title):
    """
    Click the same physical bottle feature in each image.
    Returns one point with shape (1, 2), in original pixel coordinates.
    """
    max_w, max_h = 650, 850
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    display = cv2.resize(
        image,
        (round(w * scale), round(h * scale))
    )

    state = {"point": None}
    window_name = f"Click bottle feature - {title}"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, display)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (x, y)

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n{title}: click the SAME physical bottle point.")
    print("Left-click to select; R to retry; Q or ESC to abort.")

    while True:
        # Keep the window responsive and process mouse/keyboard events.
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if state["point"] is not None:
            px, py = state["point"]

            preview = display.copy()
            cv2.circle(preview, (px, py), 6, (0, 0, 255), -1)
            cv2.imshow(window_name, preview)
            cv2.waitKey(250)

            cv2.destroyWindow(window_name)

            # Convert from resized-display coordinates to source pixels.
            display_h, display_w = display.shape[:2]

            original_x = px * (w / display_w)
            original_y = py * (h / display_h)

            return np.array(
                [[original_x, original_y]], dtype=np.float64
            )

        if key == ord("r"):
            state["point"] = None

        elif key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Point selection cancelled.")

        # If the user closes the window using its X button, abort cleanly.
        if cv2.getWindowProperty(
            window_name, cv2.WND_PROP_VISIBLE
        ) < 1:
            raise RuntimeError(
                f"Selection window closed before choosing a point: {title}"
            )


def triangulate_multiview(points_norm, projections):
    """Linear triangulation from normalized camera coordinates."""
    A = []

    for (x, y), P in zip(points_norm, projections):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]

    if abs(Xh[3]) < 1e-10:
        raise RuntimeError(
            "Triangulation failed: degenerate geometry or point at infinity."
        )

    return Xh[:3] / Xh[3]


def main():
    if not (
        len(IMAGE_FILES)
        == len(CAMERA_X_CM)
        == len(CAMERA_YAW_DEG)
    ):
        raise ValueError(
            "IMAGE_FILES, CAMERA_X_CM and CAMERA_YAW_DEG "
            "must have equal lengths."
        )

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE.resolve()}"
        )

    K, dist, calibration_size = load_calibration()

    raw_points = []

    # Load images and select corresponding physical features.
    for filename in IMAGE_FILES:
        path = IMAGE_DIR / filename
        image = cv2.imread(str(path))

        if image is None:
            raise FileNotFoundError(
                f"Cannot load image: {path.resolve()}"
            )

        h, w = image.shape[:2]

        if (w, h) != calibration_size:
            raise ValueError(
                f"{filename} size {(w, h)} does not match calibration "
                f"size {calibration_size}."
            )

        raw_points.append(click_feature(image, filename))

    cv2.destroyAllWindows()

    # Undistort pixel coordinates into normalized camera coordinates.
    points_norm = []

    for point in raw_points:
        undistorted = cv2.undistortPoints(
            point.reshape(-1, 1, 2), K, dist
        )
        points_norm.append(undistorted.reshape(2))

    centres, rotations, projections = [], [], []

    for x, yaw in zip(CAMERA_X_CM, CAMERA_YAW_DEG):
        C, R_wc, t_wc, P = camera_pose(x, yaw)
        centres.append(C)
        rotations.append(R_wc)
        projections.append(P)

    # Reconstruct the point using all three views.
    X_world = triangulate_multiview(points_norm, projections)

    print("\n========== TRIANGULATION RESULTS ==========")
    print("Reconstructed world point (cm):")
    print(f"X = {X_world[0]:.3f}")
    print(f"Y = {X_world[1]:.3f}")
    print(f"Z = {X_world[2]:.3f}")

    print("\n========== PER-CAMERA DEPTH ==========")

    for filename, R_wc, C in zip(IMAGE_FILES, rotations, centres):
        X_cam = R_wc @ (X_world - C)
        print(f"{filename}: {X_cam[2]:.3f} cm optical-axis depth")

        if X_cam[2] <= 0:
            print("WARNING: reconstructed point is behind this camera.")

    print("\n========== REPROJECTION ERRORS ==========")
    errors = []

    for i, (R_wc, C) in enumerate(zip(rotations, centres)):
        rvec, _ = cv2.Rodrigues(R_wc)
        tvec = (-R_wc @ C).reshape(3, 1)

        projected, _ = cv2.projectPoints(
            X_world.reshape(1, 1, 3),
            rvec,
            tvec,
            K,
            dist
        )

        projected_px = projected.reshape(2)
        observed_px = raw_points[i].reshape(2)
        error = float(np.linalg.norm(projected_px - observed_px))

        errors.append(error)
        print(f"{IMAGE_FILES[i]}: {error:.3f} px")

    print(f"Mean reprojection error: {np.mean(errors):.3f} px")

    estimated_depth = float(X_world[2])
    absolute_error = abs(estimated_depth - GROUND_TRUTH_DEPTH_CM)

    relative_error = (
        absolute_error / abs(GROUND_TRUTH_DEPTH_CM) * 100
        if GROUND_TRUTH_DEPTH_CM != 0 else np.nan
    )

    print("\n========== GROUND-TRUTH COMPARISON ==========")
    print(f"Estimated perpendicular depth: {estimated_depth:.3f} cm")
    print(f"Ground-truth depth:            {GROUND_TRUTH_DEPTH_CM:.3f} cm")
    print(f"Absolute error:                 {absolute_error:.3f} cm")
    print(f"Relative error:                 {relative_error:.2f}%")

    print("\n========== PAIRWISE CHECK ==========")

    for i, j in combinations(range(len(raw_points)), 2):
        X_pair = triangulate_multiview(
            [points_norm[i], points_norm[j]],
            [projections[i], projections[j]]
        )

        print(
            f"{IMAGE_FILES[i]} + {IMAGE_FILES[j]}: "
            f"X={X_pair[0]:.2f}, Y={X_pair[1]:.2f}, "
            f"Z={X_pair[2]:.2f} cm"
        )

    np.savez(
        "stereo_depth_results.npz",
        point_world_cm=X_world,
        estimated_depth_cm=estimated_depth,
        ground_truth_depth_cm=GROUND_TRUTH_DEPTH_CM,
        absolute_error_cm=absolute_error,
        relative_error_percent=relative_error,
        reprojection_errors_px=np.array(errors)
    )

    print("\nSaved results to stereo_depth_results.npz")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        cv2.destroyAllWindows()
        print(f"\nERROR: {exc}")
