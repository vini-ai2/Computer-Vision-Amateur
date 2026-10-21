
import cv2
import numpy as np
from pathlib import Path
from itertools import combinations

# ============================================================
# CONFIGURATION — EDIT THESE VALUES
# ============================================================

IMAGE_DIR = Path("bottle_images")
CALIBRATION_FILE = Path("camera_calibration.npz")

# Use the actual image filenames.
IMAGE_FILES = ["pose1.jpg", "pose2.jpg", "pose3.jpg"]

# Camera X positions in centimetres, measured from one origin.
# Replace these examples with your measured positions.
CAMERA_X_CM = [0.0, 50.0, 100.0]

# Camera yaw at each pose, in degrees.
# Positive yaw turns the optical axis toward +X under this
# script's coordinate convention. Negative yaw turns toward -X.
# Replace these examples with your measured/estimated angles.
CAMERA_YAW_DEG = [10.0, 0.0, -10.0]

# Ground-truth perpendicular bottle depth from the X baseline,
# in cm. Enter your measured value.
GROUND_TRUTH_DEPTH_CM = 68.5

# ============================================================


def load_calibration():
    data = np.load(CALIBRATION_FILE)

    K = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64)

    calibration_size = tuple(
        int(v) for v in data["image_size"]
    )

    return K, dist, calibration_size


def rotation_yaw(degrees):
    """Camera-to-world rotation for yaw around the world Y axis."""
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ], dtype=np.float64)


def camera_pose(x_cm, yaw_deg):
    """
    World coordinates:
      X: direction of camera movement
      Y: vertical
      Z: perpendicular depth from the baseline

    Camera centres are C = (X, 0, 0).
    """
    C = np.array([x_cm, 0.0, 0.0], dtype=np.float64)

    R_cw = rotation_yaw(yaw_deg)
    R_wc = R_cw.T
    t_wc = -R_wc @ C

    # Normalized-coordinate projection matrix.
    P = np.hstack((R_wc, t_wc.reshape(3, 1)))

    return C, R_wc, t_wc, P


def click_feature(image, title):
    """
    Display an image and let the user click one identifiable
    physical point on the bottle. Returns original pixel coords.
    """
    max_w, max_h = 650, 850
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    display = cv2.resize(
        image,
        (round(w * scale), round(h * scale))
    )

    state = {"point": None}
    preview = display.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (x, y)

    window = "Click bottle feature - " + title
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)

    print(f"\n{title}: click the SAME physical bottle point.")
    print("Press R to retry; press Q or ESC to abort.")

    while True:
        preview = display.copy()

        if state["point"] is not None:
            cv2.circle(
                preview, state["point"], 6, (0, 0, 255), -1
            )

        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF

        if state["point"] is not None and key in (13, 32):
            break

        if key == ord("r"):
            state["point"] = None

        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise RuntimeError("Point selection cancelled.")

        # Double-click also accepts the point.
        # Alternatively, press Enter or Space after clicking.

    cv2.destroyWindow(window)

    px, py = state["point"]
    return np.array(
        [[px / scale, py / scale]],
        dtype=np.float64
    )


def triangulate_multiview(points_norm, projections):
    """
    Linear multi-view triangulation using normalized image points.
    Each point is (x, y) in normalized camera coordinates.
    """
    A = []

    for (x, y), P in zip(points_norm, projections):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]

    if abs(Xh[3]) < 1e-10:
        raise RuntimeError(
            "Triangulation failed: point is at infinity or geometry "
            "is degenerate."
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

    K, dist, calibration_size = load_calibration()

    images = []
    raw_points = []

    # Load images and manually select corresponding bottle features.
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
                f"{filename} has size {(w, h)}, but calibration "
                f"uses {calibration_size}. Use matching resolution."
            )

        images.append(image)
        raw_points.append(click_feature(image, filename))

    # Convert observed pixels into undistorted normalized coordinates.
    points_norm = []

    for point in raw_points:
        undistorted = cv2.undistortPoints(
            point.reshape(-1, 1, 2), K, dist
        )
        points_norm.append(undistorted.reshape(2))

    # Construct world-to-camera projection matrices.
    centres = []
    rotations = []
    projections = []

    for x, yaw in zip(CAMERA_X_CM, CAMERA_YAW_DEG):
        C, R_wc, t_wc, P = camera_pose(x, yaw)

        centres.append(C)
        rotations.append(R_wc)
        projections.append(P)

    # Reconstruct one 3D point using all three views.
    X_world = triangulate_multiview(
        points_norm, projections
    )

    print("\n========== TRIANGULATION RESULTS ==========")
    print("3D bottle point in world coordinates (cm):")
    print(f"X = {X_world[0]:.3f}")
    print(f"Y = {X_world[1]:.3f}")
    print(f"Z = {X_world[2]:.3f}")

    # Check whether the reconstructed point is in front of each camera.
    print("\nDepth in each camera's coordinate frame:")

    for i, (R_wc, C) in enumerate(zip(rotations, centres)):
        X_cam = R_wc @ (X_world - C)
        print(
            f"{IMAGE_FILES[i]}: "
            f"optical-axis depth = {X_cam[2]:.3f} cm"
        )

        if X_cam[2] <= 0:
            print("  WARNING: point is behind this camera.")

    # Reprojection error: project the 3D point back into each image.
    errors = []

    for i, (R_wc, C) in enumerate(zip(rotations, centres)):
        X_cam = R_wc @ (X_world - C)

        if X_cam[2] <= 0:
            errors.append(np.nan)
            continue

        projected, _ = cv2.projectPoints(
            X_world.reshape(1, 1, 3),
            cv2.Rodrigues(R_wc)[0],
            -R_wc @ C,
            K,
            dist
        )

        projected_px = projected.reshape(2)
        observed_px = raw_points[i].reshape(2)

        error = np.linalg.norm(projected_px - observed_px)
        errors.append(error)

        print(
            f"Reprojection error, {IMAGE_FILES[i]}: "
            f"{error:.3f} px"
        )

    print(
        "Mean reprojection error:",
        f"{np.nanmean(errors):.3f} px"
    )

    # Z is perpendicular depth from the baseline in this coordinate frame.
    estimated_depth = X_world[2]

    print("\n========== GROUND-TRUTH COMPARISON ==========")
    print(f"Estimated perpendicular depth: {estimated_depth:.3f} cm")
    print(
        f"Ground-truth depth:            "
        f"{GROUND_TRUTH_DEPTH_CM:.3f} cm"
    )

    absolute_error = abs(
        estimated_depth - GROUND_TRUTH_DEPTH_CM
    )

    relative_error = (
        absolute_error / abs(GROUND_TRUTH_DEPTH_CM) * 100
        if GROUND_TRUTH_DEPTH_CM != 0 else np.nan
    )

    print(f"Absolute error:                 {absolute_error:.3f} cm")
    print(f"Relative error:                 {relative_error:.2f} %")

    # Pairwise triangulations help reveal inconsistent views.
    print("\n========== PAIRWISE CHECK ==========")

    for i, j in combinations(range(len(images)), 2):
        X_pair = triangulate_multiview(
            [points_norm[i], points_norm[j]],
            [projections[i], projections[j]]
        )

        print(
            f"{IMAGE_FILES[i]} + {IMAGE_FILES[j]}: "
            f"X={X_pair[0]:.2f}, "
            f"Y={X_pair[1]:.2f}, "
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
    main()
