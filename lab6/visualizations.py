import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

IMAGE_DIR = Path("bottle_images")
CALIBRATION_FILE = Path("camera_calibration.npz")
IMAGE_FILES = ["pose1.jpeg", "pose2.jpeg", "pose3.jpeg"]
CAMERA_X_CM = [0.0, 50.0, 100.0]
CAMERA_YAW_DEG = [31.0, 0.0, -31.0]
GROUND_TRUTH_DEPTH_CM = 68.5


def load_calibration():
    data = np.load(CALIBRATION_FILE)
    return (
        data["camera_matrix"].astype(np.float64),
        data["dist_coeffs"].astype(np.float64),
        tuple(int(v) for v in data["image_size"]),
    )


def rotation_yaw(degrees):
    """Camera-to-world rotation for yaw around world Y."""
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def camera_pose(x_cm, yaw_deg):
    C = np.array([x_cm, 0.0, 0.0], dtype=np.float64)
    R_cw = rotation_yaw(yaw_deg)
    R_wc = R_cw.T
    t_wc = -R_wc @ C
    P = np.hstack((R_wc, t_wc.reshape(3, 1)))
    return C, R_wc, t_wc, P


def click_feature(image, title):
    """Click the same physical bottle feature; return original pixel coordinates."""
    max_w, max_h = 650, 850
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    display = cv2.resize(image, (round(w * scale), round(h * scale)))
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
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if state["point"] is not None:
            px, py = state["point"]
            preview = display.copy()
            cv2.circle(preview, (px, py), 6, (0, 0, 255), -1)
            cv2.imshow(window_name, preview)
            cv2.waitKey(250)
            cv2.destroyWindow(window_name)

            display_h, display_w = display.shape[:2]
            original_x = px * (w / display_w)
            original_y = py * (h / display_h)
            return np.array([[original_x, original_y]], dtype=np.float64)

        if key == ord("r"):
            state["point"] = None
        elif key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Point selection cancelled.")

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError(
                    f"Selection window closed before choosing a point: {title}"
                )
        except cv2.error:
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
        raise RuntimeError("Triangulation failed: degenerate geometry or point at infinity.")
    return Xh[:3] / Xh[3]



def create_visualizations(images, raw_points, X_world, centres, rotations,
                          errors, pairwise_depths, projected_points, K, dist):
    """Generate diagnostic plots and dense stereo depth maps."""
    out = Path("visualizations")
    out.mkdir(exist_ok=True)
    gt = GROUND_TRUTH_DEPTH_CM
    estimate = float(X_world[2])

    # 1. Camera geometry
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    C_arr = np.asarray(centres)
    ax.plot(C_arr[:, 0], C_arr[:, 1], C_arr[:, 2], "o-",
            label="Camera centres")

    for i, (C, R_wc) in enumerate(zip(centres, rotations)):
        direction = R_wc.T[:, 2]
        ax.quiver(*C, *direction, length=25, normalize=True)
        ax.text(*C, f" Pose {i + 1}")

    ax.scatter(*X_world, marker="*", s=180, label="Triangulated point")
    ax.set_xlabel("World X (cm)")
    ax.set_ylabel("World Y (cm)")
    ax.set_zlabel("World Z (cm)")
    ax.set_title("Moving-camera stereo geometry")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_camera_geometry.png", dpi=180)
    plt.close(fig)

    # 2. Observed vs reprojected points
    fig, axes = plt.subplots(1, len(images), figsize=(16, 8))
    if len(images) == 1:
        axes = [axes]

    for i, (image, observed, predicted) in enumerate(
            zip(images, raw_points, projected_points)):
        axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ox, oy = observed.reshape(2)
        px, py = predicted.reshape(2)
        axes[i].scatter(ox, oy, c="lime", s=80, label="Observed")
        axes[i].scatter(px, py, c="red", marker="x", s=90,
                        label="Reprojected")
        axes[i].plot([ox, px], [oy, py], color="yellow")
        axes[i].set_title(f"{IMAGE_FILES[i]}\nError: {errors[i]:.1f} px")
        axes[i].axis("off")
        axes[i].legend()

    fig.tight_layout()
    fig.savefig(out / "02_reprojection.png", dpi=180)
    plt.close(fig)

    # 3. Pairwise depth consistency
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(pairwise_depths.keys())
    values = list(pairwise_depths.values())
    ax.bar(labels, values)
    ax.axhline(gt, linestyle="--", label=f"Ground truth: {gt:.1f} cm")
    ax.axhline(estimate, linestyle=":",
               label=f"Three-view estimate: {estimate:.2f} cm")
    ax.set_ylabel("Depth (cm)")
    ax.set_title("Pairwise triangulation consistency")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_pairwise_depth.png", dpi=180)
    plt.close(fig)

    # 4. Reprojection errors
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(IMAGE_FILES, errors)
    ax.set_ylabel("Reprojection error (pixels)")
    ax.set_title("Reprojection errors")
    fig.tight_layout()
    fig.savefig(out / "04_reprojection_errors.png", dpi=180)
    plt.close(fig)

    # 5. Dense depth maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (i, j) in zip(axes, combinations(range(len(images)), 2)):
        # Relative pose of camera j with respect to camera i
        R = rotations[j] @ rotations[i].T
        T = (rotations[j] @ (centres[i] - centres[j])).reshape(3, 1)

        h, w = images[i].shape[:2]
        size = (w, h)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, dist, K, dist, size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY
        )

        map1x, map1y = cv2.initUndistortRectifyMap(
            K, dist, R1, P1, size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            K, dist, R2, P2, size, cv2.CV_32FC1
        )

        left = cv2.remap(images[i], map1x, map1y, cv2.INTER_LINEAR)
        right = cv2.remap(images[j], map2x, map2y, cv2.INTER_LINEAR)

        # Quarter resolution limits memory use
        scale = 0.25
        match_size = (w // 4, h // 4)

        gray_left = cv2.resize(
            cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
            match_size, interpolation=cv2.INTER_AREA
        )
        gray_right = cv2.resize(
            cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
            match_size, interpolation=cv2.INTER_AREA
        )

        # Keep disparity range valid for the resized image width
        num_disp = min(128, ((match_size[0] - 32) // 16) * 16)
        if num_disp < 16:
            ax.set_title("Image too small for stereo matching")
            continue

        min_disp = -num_disp if P2[0, 3] > 0 else 0
        block = 5

        matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * block**2,
            P2=32 * block**2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=0,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM
        )

        disparity = matcher.compute(
            gray_left, gray_right
        ).astype(np.float32) / 16.0

        # Convert resized-image coordinates back to original scale
        Q_scaled = Q @ np.diag([
            1 / scale, 1 / scale, 1 / scale, 1
        ])

        xyz = cv2.reprojectImageTo3D(disparity, Q_scaled)
        depth = xyz[:, :, 2]

        valid = (
            (disparity > min_disp)
            & np.isfinite(depth)
            & (depth > 0)
            & (depth < 200)
        )

        # Prefer the teal bottle, but do not force an empty mask
        hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
        bottle_mask = cv2.inRange(
            hsv, np.array([35, 20, 20]),
            np.array([115, 255, 255])
        )
        bottle_mask = cv2.resize(
            bottle_mask, match_size,
            interpolation=cv2.INTER_NEAREST
        ) > 0

        if np.count_nonzero(valid & bottle_mask) >= 20:
            valid &= bottle_mask

        depth_plot = np.full(depth.shape, np.nan, dtype=np.float32)
        depth_plot[valid] = depth[valid]

        print(
            f"{IMAGE_FILES[i]} + {IMAGE_FILES[j]}: "
            f"{np.count_nonzero(valid)} valid pixels"
        )

        plot = ax.imshow(depth_plot, cmap="turbo", vmin=0, vmax=200)
        ax.set_title(f"{IMAGE_FILES[i]} + {IMAGE_FILES[j]}")
        ax.axis("off")
        fig.colorbar(plot, ax=ax, label="Depth (cm)")

    fig.tight_layout()
    fig.savefig(out / "05_depth_maps.png", dpi=180)
    plt.close(fig)

    print(f"Visualizations saved in: {out.resolve()}")


def main():
    if not (len(IMAGE_FILES) == len(CAMERA_X_CM) == len(CAMERA_YAW_DEG)):
        raise ValueError("IMAGE_FILES, CAMERA_X_CM and CAMERA_YAW_DEG must have equal lengths.")
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_FILE.resolve()}")

    K, dist, calibration_size = load_calibration()
    images, raw_points = [], []

    for filename in IMAGE_FILES:
        path = IMAGE_DIR / filename
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {path.resolve()}")
        h, w = image.shape[:2]
        if (w, h) != calibration_size:
            raise ValueError(
                f"{filename} size {(w, h)} does not match calibration size {calibration_size}."
            )
        images.append(image)
        raw_points.append(click_feature(image, filename))

    cv2.destroyAllWindows()

    points_norm = []
    for point in raw_points:
        undistorted = cv2.undistortPoints(point.reshape(-1, 1, 2), K, dist)
        points_norm.append(undistorted.reshape(2))

    centres, rotations, projections = [], [], []
    for x, yaw in zip(CAMERA_X_CM, CAMERA_YAW_DEG):
        C, R_wc, t_wc, P = camera_pose(x, yaw)
        centres.append(C)
        rotations.append(R_wc)
        projections.append(P)

    X_world = triangulate_multiview(points_norm, projections)
    print("\\n========== TRIANGULATION RESULTS ==========")
    print("Reconstructed world point (cm):")
    print(f"X = {X_world[0]:.3f}\\nY = {X_world[1]:.3f}\\nZ = {X_world[2]:.3f}")

    print("\\n========== PER-CAMERA DEPTH ==========")
    for filename, R_wc, C in zip(IMAGE_FILES, rotations, centres):
        X_cam = R_wc @ (X_world - C)
        print(f"{filename}: {X_cam[2]:.3f} cm optical-axis depth")
        if X_cam[2] <= 0:
            print("WARNING: reconstructed point is behind this camera.")

    print("\\n========== REPROJECTION ERRORS ==========")
    errors, projected_points = [], []
    for i, (R_wc, C) in enumerate(zip(rotations, centres)):
        rvec, _ = cv2.Rodrigues(R_wc)
        tvec = (-R_wc @ C).reshape(3, 1)
        projected, _ = cv2.projectPoints(X_world.reshape(1, 1, 3), rvec, tvec, K, dist)
        projected_px = projected.reshape(2)
        projected_points.append(projected_px.reshape(1, 2))
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
    print("\\n========== GROUND-TRUTH COMPARISON ==========")
    print(f"Estimated perpendicular depth: {estimated_depth:.3f} cm")
    print(f"Ground-truth depth:            {GROUND_TRUTH_DEPTH_CM:.3f} cm")
    print(f"Absolute error:                {absolute_error:.3f} cm")
    print(f"Relative error:                {relative_error:.2f}%")

    print("\\n========== PAIRWISE CHECK ==========")
    pairwise_depths = {}
    for i, j in combinations(range(len(raw_points)), 2):
        X_pair = triangulate_multiview(
            [points_norm[i], points_norm[j]], [projections[i], projections[j]]
        )
        label = f"{IMAGE_FILES[i]} + {IMAGE_FILES[j]}"
        pairwise_depths[label] = float(X_pair[2])
        print(f"{label}: X={X_pair[0]:.2f}, Y={X_pair[1]:.2f}, Z={X_pair[2]:.2f} cm")

    np.savez(
        "stereo_depth_results.npz",
        point_world_cm=X_world,
        estimated_depth_cm=estimated_depth,
        ground_truth_depth_cm=GROUND_TRUTH_DEPTH_CM,
        absolute_error_cm=absolute_error,
        relative_error_percent=relative_error,
        reprojection_errors_px=np.array(errors),
        observed_points_px=np.array(raw_points).reshape(-1, 2),
        projected_points_px=np.array(projected_points).reshape(-1, 2),
        camera_centres_cm=np.array(centres),
        camera_yaw_deg=np.array(CAMERA_YAW_DEG),
        pairwise_depths_cm=np.array(list(pairwise_depths.values())),
    )
    print("\\nSaved results to stereo_depth_results.npz")

    create_visualizations(
        images, raw_points, X_world, centres, rotations,
        errors, pairwise_depths, projected_points, K, dist
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        cv2.destroyAllWindows()
        print(f"\\nERROR: {exc}")
