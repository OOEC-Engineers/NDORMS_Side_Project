#!/usr/bin/env python3
"""
2D-3D Knee Registration using DiffDRR + PyTorch Gradient Descent.

Registers a 3D STL model (knee bone) to a 2D radiograph by optimising
6-DoF rigid pose (3 Euler rotations + 3 translations) to maximise
Normalised Cross-Correlation (NCC) between a synthesised DRR and the
radiograph ROI.

Usage (interactive ROI):
    python KneeRegistration2D3D.py \
        --stl 3D_files/Phase3FemurM.stl \
        --xray Xray/ASIP19.jpg \
        --sdd 1024 --pixel_spacing 0.25

Usage (explicit ROI):
    python KneeRegistration2D3D.py \
        --stl 3D_files/Phase3FemurM.stl \
        --xray Xray/ASIP19.jpg \
        --roi 200 150 400 500 \
        --sdd 1024 --pixel_spacing 0.25
"""

import sys
import os
import json
import argparse
import time


# ---------------------------------------------------------------------------
# Dependency checks — fail fast with helpful messages
# ---------------------------------------------------------------------------

def _check_deps():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import trimesh
    except ImportError:
        missing.append("trimesh")
    try:
        import torchio
    except ImportError:
        missing.append("torchio  (pip install torchio)")
    try:
        import diffdrr
    except ImportError:
        missing.append("diffdrr  (pip install diffdrr)")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import PIL
    except ImportError:
        missing.append("Pillow  (pip install Pillow)")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python  (pip install opencv-python)")
    if missing:
        print("[ERROR] Missing dependencies:")
        for m in missing:
            print(f"         - {m}")
        print("\nInstall all at once:")
        print("  pip install torch trimesh torchio diffdrr numpy Pillow matplotlib opencv-python scipy")
        sys.exit(1)


_check_deps()

import numpy as np
import torch
import trimesh
import torchio as tio
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import cv2
from scipy.ndimage import binary_fill_holes


# ---------------------------------------------------------------------------
# 1.  STL -> Voxel Volume
# ---------------------------------------------------------------------------

def load_and_voxelize(stl_path, voxel_size=1.0, stl_scale=1.0):
    """Load STL mesh and voxelize into a dense 3D volume.

    Returns:
        volume     : torch.Tensor of shape (1, X, Y, Z), float32 density
        affine     : 4x4 voxel-index-to-STL transform in millimetres
        stl_center : centre used to place the STL at the DiffDRR world origin
    """
    print(f"[INFO] Loading STL: {stl_path}")
    mesh = trimesh.load(stl_path, force="mesh")
    print(f"[INFO] Mesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
    if stl_scale != 1.0:
        mesh.apply_scale(stl_scale)
        print(f"[INFO] Applied uniform STL scale factor: {stl_scale}")

    # Voxelize the mesh surface, then fill interior
    voxel_grid = mesh.voxelized(voxel_size).fill()
    affine = np.asarray(voxel_grid.transform, dtype=np.float64)
    voxel_matrix = voxel_grid.matrix  # boolean numpy array (X, Y, Z)

    # Fill any remaining holes
    volume_np = binary_fill_holes(voxel_matrix).astype(np.float32)

    # TorchIO uses (channels, X, Y, Z), matching trimesh's voxel matrix.
    volume = torch.from_numpy(volume_np).float().unsqueeze(0)

    image_center_index = (np.asarray(volume_np.shape, dtype=np.float64) - 1.0) / 2.0
    stl_center = (affine @ np.r_[image_center_index, 1.0])[:3]

    print(f"[INFO] Mesh bounds: {mesh.bounds}")
    print(f"[INFO] Voxel volume shape: {volume.shape}, spacing: {voxel_size} mm")
    print(f"[INFO] STL centre used by DiffDRR: {stl_center} mm")

    return volume, affine, stl_center


# ---------------------------------------------------------------------------
# 2.  Radiograph loading + ROI selection
# ---------------------------------------------------------------------------

def load_image_as_array(path):
    """Load radiograph as grayscale numpy array (H, W) float32 in [0, 255]."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def detect_scale_endpoints(img_array):
    """Estimate the endpoints of a long, bright, axis-aligned X-ray ruler."""
    image = np.clip(img_array, 0, 255).astype(np.uint8)
    height, width = image.shape
    edges = cv2.Canny(image, 80, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(30, int(0.04 * max(height, width))),
        minLineLength=max(40, int(0.10 * max(height, width))),
        maxLineGap=max(10, int(0.03 * max(height, width))),
    )

    best = None
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines, dtype=np.int32).reshape(-1, 4):
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length <= 0.0:
                continue
            axis_alignment = max(abs(dx), abs(dy)) / length
            if axis_alignment < 0.96:
                continue

            sample_count = max(2, int(length))
            sample_x = np.linspace(x1, x2, sample_count).round().astype(int)
            sample_y = np.linspace(y1, y2, sample_count).round().astype(int)
            sample_x = np.clip(sample_x, 0, width - 1)
            sample_y = np.clip(sample_y, 0, height - 1)
            brightness = float(np.percentile(image[sample_y, sample_x], 75))

            midpoint_x = (x1 + x2) / 2.0
            midpoint_y = (y1 + y2) / 2.0
            border_distance = min(
                midpoint_x,
                width - 1 - midpoint_x,
                midpoint_y,
                height - 1 - midpoint_y,
            )
            border_bonus = 1.0 + max(
                0.0,
                1.0 - border_distance / (0.30 * min(height, width)),
            )
            brightness_weight = 0.25 + 0.75 * brightness / 255.0
            score = length * axis_alignment * brightness_weight * border_bonus
            if best is None or score > best[0]:
                best = (score, np.array([[x1, y1], [x2, y2]], dtype=np.float64))

    if best is None:
        print("[WARN] Could not detect an X-ray ruler; using editable fallback points.")
        return np.array(
            [[0.10 * width, 0.25 * height], [0.10 * width, 0.75 * height]],
            dtype=np.float64,
        )

    points = best[1]
    delta = np.abs(points[1] - points[0])
    order_axis = 1 if delta[1] >= delta[0] else 0
    points = points[np.argsort(points[:, order_axis])]
    print(
        "[INFO] Automatically detected scale endpoints: "
        f"({points[0, 0]:.1f}, {points[0, 1]:.1f}) and "
        f"({points[1, 0]:.1f}, {points[1, 1]:.1f})"
    )
    return points


def adjust_scale_points_interactive(img_array, initial_points, known_length_cm):
    """Let the user drag auto-detected scale points with zoom and pan."""
    points = np.asarray(initial_points, dtype=np.float64).reshape(2, 2).copy()
    height, width = img_array.shape

    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    ax.imshow(img_array, cmap="gray")
    line, = ax.plot(
        points[:, 0], points[:, 1], "r-", linewidth=2.0, zorder=3
    )
    markers = ax.scatter(
        points[:, 0], points[:, 1],
        s=110, c="red", edgecolors="yellow", linewidths=1.5, zorder=4,
    )
    info = ax.text(
        0.01, 0.01, "", transform=ax.transAxes,
        color="yellow", fontsize=10, va="bottom",
        bbox=dict(facecolor="black", alpha=0.70, edgecolor="none"),
        zorder=5,
    )
    ax.set_title(
        f"Adjust the two endpoints spanning {known_length_cm:g} cm\n"
        "Left-drag point | Wheel zoom | Right-drag pan | R reset | Enter accept"
    )
    ax.axis("off")
    original_xlim = (-0.5, width - 0.5)
    original_ylim = (height - 0.5, -0.5)
    ax.set_xlim(original_xlim)
    ax.set_ylim(original_ylim)

    state = {"drag": None, "pan": None}

    def refresh():
        line.set_data(points[:, 0], points[:, 1])
        markers.set_offsets(points)
        distance = float(np.linalg.norm(points[1] - points[0]))
        spacing = known_length_cm * 10.0 / max(distance, 1e-12)
        info.set_text(
            f"Distance: {distance:.2f} px\nScale: {spacing:.6f} mm/px"
        )
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes is not ax:
            return
        if event.button == 1:
            display_points = ax.transData.transform(points)
            distances = np.linalg.norm(
                display_points - np.array([event.x, event.y]), axis=1
            )
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 18.0:
                state["drag"] = nearest
        elif event.button in (2, 3):
            state["pan"] = (
                event.x, event.y, ax.get_xlim(), ax.get_ylim()
            )

    def on_motion(event):
        if event.inaxes is not ax:
            return
        if state["drag"] is not None and event.xdata is not None and event.ydata is not None:
            points[state["drag"]] = [event.xdata, event.ydata]
            refresh()
        elif state["pan"] is not None:
            start_x, start_y, xlim, ylim = state["pan"]
            dx = (event.x - start_x) * (xlim[1] - xlim[0]) / ax.bbox.width
            dy = (event.y - start_y) * (ylim[1] - ylim[0]) / ax.bbox.height
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            fig.canvas.draw_idle()

    def on_release(event):
        state["drag"] = None
        state["pan"] = None

    def on_scroll(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        factor = 0.75 if event.button == "up" else 1.0 / 0.75
        xlim = np.asarray(ax.get_xlim(), dtype=np.float64)
        ylim = np.asarray(ax.get_ylim(), dtype=np.float64)
        ax.set_xlim(event.xdata + (xlim - event.xdata) * factor)
        ax.set_ylim(event.ydata + (ylim - event.ydata) * factor)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("enter", "return"):
            plt.close(fig)
        elif event.key in ("r", "R"):
            ax.set_xlim(original_xlim)
            ax.set_ylim(original_ylim)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()
    return points


def calibrate_pixel_spacing(img_array, known_length_cm, out_dir, points=None):
    """Calibrate detector spacing by selecting a known ruler length."""
    if points is None:
        print(
            f"[INFO] Scale calibration: checking automatically placed endpoints "
            f"for the {known_length_cm:g} cm reference."
        )
        detected_points = detect_scale_endpoints(img_array)
        points = adjust_scale_points_interactive(
            img_array, detected_points, known_length_cm
        )
    else:
        points = np.asarray(points, dtype=np.float64).reshape(2, 2)

    pixel_distance = float(np.linalg.norm(points[1] - points[0]))
    if pixel_distance <= 0.0:
        raise ValueError("Scale calibration points must be different")

    physical_length_mm = float(known_length_cm) * 10.0
    pixel_spacing = physical_length_mm / pixel_distance
    print(
        f"[INFO] Scale calibration: {physical_length_mm:.3f} mm / "
        f"{pixel_distance:.3f} px = {pixel_spacing:.6f} mm/px"
    )

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_array, cmap="gray")
    ax.plot(points[:, 0], points[:, 1], "r-o", linewidth=2, markersize=6)
    midpoint = points.mean(axis=0)
    ax.text(
        midpoint[0], midpoint[1],
        f"  {known_length_cm:g} cm = {pixel_distance:.1f} px\n"
        f"  {pixel_spacing:.6f} mm/px",
        color="yellow",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
    )
    ax.set_title("X-ray Scale Calibration")
    ax.axis("off")
    calibration_image = os.path.join(out_dir, "scale_calibration.png")
    fig.savefig(calibration_image, dpi=150, bbox_inches="tight")
    plt.close(fig)

    calibration_data = {
        "known_length_cm": float(known_length_cm),
        "known_length_mm": physical_length_mm,
        "points_xy_pixels": points.tolist(),
        "pixel_distance": pixel_distance,
        "pixel_spacing_mm": pixel_spacing,
    }
    calibration_json = os.path.join(out_dir, "scale_calibration.json")
    with open(calibration_json, "w") as f:
        json.dump(calibration_data, f, indent=2)
    print(f"[INFO] Saved scale calibration: {calibration_image}")
    print(f"[INFO] Saved scale calibration data: {calibration_json}")
    return pixel_spacing


def interactive_roi_select(img_array):
    """Open a matplotlib window and let the user draw a bounding box.

    Returns (x, y, w, h) in pixel coordinates.
    """
    print("[INFO] Interactive ROI selection - draw a box on the image, then close the window.")

    # Try to use an interactive backend
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            try:
                matplotlib.use("MacOSX")
            except Exception:
                print("[ERROR] No interactive matplotlib backend available.")
                print("        Please install tkinter (often: brew install python-tk)")
                print("        Or pass --roi X Y W H on the command line.")
                sys.exit(1)

    # Re-import pyplot after backend switch
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    selected = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_array, cmap="gray")
    ax.set_title("Draw a rectangle around the bone region, then close the window.")
    ax.axis("off")

    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        x = int(min(x1, x2))
        y = int(min(y1, y2))
        w = int(abs(x2 - x1))
        h = int(abs(y2 - y1))
        selected.clear()
        selected.append((x, y, w, h))
        print(f"[INFO] Selected ROI: x={x}, y={y}, w={w}, h={h}")

    # NOTE: 'rectprops' was renamed to 'props' in matplotlib >= 3.5
    rect_selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        props=dict(edgecolor="red", facecolor="red", alpha=0.3, linewidth=2),
    )

    plt.show()

    if not selected:
        print("[ERROR] No ROI was selected. Please try again or use --roi on the command line.")
        sys.exit(1)

    return selected[0]



def preprocess_roi(roi_img):
    """Enhance contrast and normalise the ROI to [0, 1]."""
    roi_uint8 = np.clip(roi_img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi_uint8)
    normalised = enhanced.astype(np.float32) / 255.0
    return normalised


def segment_implant_bbox(roi_img, percentile=80.0):
    """Segment a bright implant/component and return its bounding box and mask."""
    image = np.clip(roi_img, 0, 255).astype(np.uint8)
    threshold = float(np.percentile(image, percentile))
    mask = (image >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise RuntimeError("Could not segment a bright implant in the ROI")

    height, width = image.shape
    image_center = np.array([width / 2.0, height / 2.0])
    image_diagonal = float(np.hypot(width, height))
    best_label = None
    best_score = -float("inf")
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        if area < 0.005 * image.size:
            continue
        center_distance = np.linalg.norm(centroids[label] - image_center)
        centrality = np.exp(-2.0 * (center_distance / image_diagonal) ** 2)
        elongation_bonus = np.sqrt(max(box_width * box_height, 1))
        score = float(area) * centrality * elongation_bonus
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        raise RuntimeError("No sufficiently large implant component was segmented")

    x, y, box_width, box_height, _ = stats[best_label]
    selected_mask = (labels == best_label).astype(np.uint8)
    bbox = (int(x), int(y), int(box_width), int(box_height))
    print(
        f"[INFO] Segmented implant bbox: x={bbox[0]}, y={bbox[1]}, "
        f"w={bbox[2]}, h={bbox[3]} (percentile={percentile:g})"
    )
    return bbox, selected_mask


def segment_bone_bbox(roi_img):
    """Segment bone (mid-gray tissue) using Otsu thresholding on CLAHE ROI.

    Unlike segment_implant_bbox which targets the brightest pixels (metal
    implants), this adaptively finds the bone / soft-tissue boundary -- the
    correct target for non-implant X-ray registration.

    Returns (x, y, w, h) bounding box and a binary mask.
    """
    image = np.clip(roi_img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Otsu finds the bone / soft-tissue split automatically -- no percentile guess.
    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Clean up: close gaps, remove small blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        raise RuntimeError("No bone region found with Otsu thresholding")

    # Pick the largest non-background component (the bone).
    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, box_width, box_height = stats[best_label, :4]
    selected_mask = (labels == best_label).astype(np.uint8)
    bbox = (int(x), int(y), int(box_width), int(box_height))
    print(
        f"[INFO] Segmented bone bbox (Otsu): x={bbox[0]}, y={bbox[1]}, "
        f"w={bbox[2]}, h={bbox[3]}"
    )
    return bbox, selected_mask


def closed_form_ty(stl_path, stl_scale, target_box, pixel_spacing, sdd):
    """Compute ty directly from the magnification ratio  M = projected / real.

    In DiffDRR's convention, ty is the source-to-isocenter distance and
    magnification M = sdd / ty.  Rearranging:  ty = sdd / M.

    This eliminates the need for a brute-force depth sweep -- one STL
    measurement plus one bone-bbox measurement gives an exact initial depth.
    """
    mesh = trimesh.load(stl_path, force="mesh")
    if stl_scale != 1.0:
        mesh.apply_scale(stl_scale)
    extents = mesh.bounding_box.extents  # (dx, dy, dz) in mm
    # Use the largest beam-perpendicular extent for robustness.
    real_size_mm = float(max(extents))

    projected_size_mm = (
        max(target_box["long_side_px"], target_box["short_side_px"])
        * pixel_spacing
    )

    if projected_size_mm <= 0:
        raise ValueError("Projected bone size must be positive")

    magnification = projected_size_mm / real_size_mm
    if magnification <= 0:
        raise ValueError(f"Invalid magnification {magnification}")

    ty = sdd / magnification
    print(
        f"[INFO] Closed-form ty (oriented target box): "
        f"real={real_size_mm:.1f}mm "
        f"proj={projected_size_mm:.1f}mm "
        f"({target_box['long_side_px']:.1f} x {target_box['short_side_px']:.1f} px) "
        f"M={magnification:.3f} "
        f"-> ty={ty:.2f}mm"
    )
    return ty


def _mask_bbox(mask):
    """Return (x, y, width, height) for a non-empty binary mask."""
    rows, columns = np.where(mask)
    if not len(columns):
        return None
    x0, x1 = int(columns.min()), int(columns.max())
    y0, y1 = int(rows.min()), int(rows.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _box_from_bbox(bbox):
    """Create a box descriptor from an axis-aligned bounding box."""
    x, y, box_width, box_height = bbox
    corners = np.array(
        [
            [x, y],
            [x + box_width, y],
            [x + box_width, y + box_height],
            [x, y + box_height],
        ],
        dtype=np.float64,
    )
    return {
        "bbox_xywh": (int(x), int(y), int(box_width), int(box_height)),
        "center_xy": np.array(
            [x + box_width / 2.0, y + box_height / 2.0],
            dtype=np.float64,
        ),
        "corners_xy": corners,
        "long_side_px": float(max(box_width, box_height)),
        "short_side_px": float(min(box_width, box_height)),
        "angle_deg": 0.0,
    }


def _largest_mask_contour(mask):
    """Return the largest external contour of a binary mask."""
    mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
    contour_result = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _mask_box(mask):
    """Return an oriented box descriptor for a binary mask."""
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None

    contour = _largest_mask_contour(mask)
    if contour is None or len(contour) < 3:
        return _box_from_bbox(bbox)

    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (box_width, box_height), _ = rect
    if box_width <= 1e-6 or box_height <= 1e-6:
        return _box_from_bbox(bbox)

    corners = cv2.boxPoints(rect).astype(np.float64)
    edge_vectors = np.roll(corners, -1, axis=0) - corners
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    long_edge_index = int(np.argmax(edge_lengths))
    long_edge = edge_vectors[long_edge_index]
    angle_deg = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))

    return {
        "bbox_xywh": tuple(int(value) for value in bbox),
        "center_xy": np.array([center_x, center_y], dtype=np.float64),
        "corners_xy": corners,
        "long_side_px": float(max(box_width, box_height)),
        "short_side_px": float(min(box_width, box_height)),
        "angle_deg": angle_deg,
    }


def _serialise_box(box):
    """Convert a box descriptor into JSON-friendly builtins."""
    return {
        "bbox_xywh": [int(value) for value in box["bbox_xywh"]],
        "center_xy": [float(value) for value in box["center_xy"]],
        "corners_xy": [
            [float(coord) for coord in corner] for corner in box["corners_xy"]
        ],
        "long_side_px": float(box["long_side_px"]),
        "short_side_px": float(box["short_side_px"]),
        "angle_deg": float(box["angle_deg"]),
    }


def build_segmentation_diagnostic(roi_img, segment_mode, implant_percentile):
    """Return the target mask/box used by the selected depth segmenter."""
    if segment_mode == "bone":
        bbox, mask = segment_bone_bbox(roi_img)
        label = "bone / Otsu"
    else:
        bbox, mask = segment_implant_bbox(
            roi_img, percentile=implant_percentile
        )
        label = f"implant / percentile {implant_percentile:g}"

    box = _mask_box(mask)
    if box is None:
        box = _box_from_bbox(bbox)

    return {
        "mode": str(segment_mode),
        "label": label,
        "implant_percentile": float(implant_percentile),
        "bbox_xywh": tuple(int(value) for value in bbox),
        "box": box,
        "mask": mask.astype(np.uint8),
        "mask_area_px": int(np.count_nonzero(mask)),
        "roi_area_px": int(mask.size),
    }


def _draw_box(ax, box, color, linewidth=2.0, linestyle="-", label=None):
    """Draw a box descriptor as a quadrilateral on a matplotlib axes."""
    corners = np.asarray(box["corners_xy"], dtype=np.float64)
    closed = np.vstack((corners, corners[:1]))
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    if label is not None:
        center_x, center_y = box["center_xy"]
        ax.text(
            center_x,
            center_y,
            label,
            color=color,
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"),
        )


def _draw_segmentation_diagnostic(ax, target_np, segmentation_diagnostic):
    """Draw the selected segment_mode mask on top of the radiograph ROI."""
    ax.imshow(target_np, cmap="gray")
    if segmentation_diagnostic is None:
        ax.text(
            0.5,
            0.5,
            "No segmentation\ndiagnostic",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="yellow",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
        )
        ax.set_title("Depth target segmentation")
        ax.axis("off")
        return

    mask = np.asarray(segmentation_diagnostic["mask"], dtype=bool)
    overlay = np.ma.masked_where(~mask, mask.astype(float))
    ax.imshow(overlay, cmap="spring", alpha=0.45, vmin=0.0, vmax=1.0)
    _draw_box(
        ax,
        segmentation_diagnostic["box"],
        color="lime",
        linewidth=2.2,
        label="target",
    )

    mode = segmentation_diagnostic["mode"]
    area = segmentation_diagnostic["mask_area_px"]
    roi_area = max(segmentation_diagnostic["roi_area_px"], 1)
    area_pct = 100.0 * area / roi_area
    if mode == "implant":
        title = (
            "Depth target segmentation\n"
            f"mode=implant, percentile={segmentation_diagnostic['implant_percentile']:g}"
        )
    else:
        title = "Depth target segmentation\nmode=bone / Otsu"
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        f"mask={area_pct:.1f}% ROI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.60, edgecolor="none"),
    )
    ax.axis("off")


def _projected_silhouette_mask_and_box(drr_display):
    """Return a binary projected-STL silhouette mask and its oriented box."""
    drr_array = np.asarray(drr_display, dtype=np.float32)
    finite = np.isfinite(drr_array)
    if not finite.any():
        empty = np.zeros_like(drr_array, dtype=bool)
        return empty, None

    maximum = float(np.nanmax(drr_array))
    if maximum <= 0.0:
        empty = np.zeros_like(drr_array, dtype=bool)
        return empty, None

    mask = drr_array > max(0.5 * maximum, 1e-6)
    return mask, _mask_box(mask)


def _box_comparison_metrics(projected_box, projected_mask, segmentation_diagnostic):
    """Compare final projected STL silhouette against the depth target mask/box."""
    if projected_box is None or segmentation_diagnostic is None:
        return None

    target_box = segmentation_diagnostic["box"]
    target_mask = np.asarray(segmentation_diagnostic["mask"], dtype=bool)
    projected_mask = np.asarray(projected_mask, dtype=bool)
    target_long = max(float(target_box["long_side_px"]), 1e-6)
    target_short = max(float(target_box["short_side_px"]), 1e-6)
    projected_long = float(projected_box["long_side_px"])
    projected_short = float(projected_box["short_side_px"])
    target_center = np.asarray(target_box["center_xy"], dtype=np.float64)
    projected_center = np.asarray(projected_box["center_xy"], dtype=np.float64)
    intersection = target_mask & projected_mask
    union = target_mask | projected_mask
    target_area = int(np.count_nonzero(target_mask))
    projected_area = int(np.count_nonzero(projected_mask))
    intersection_area = int(np.count_nonzero(intersection))
    union_area = int(np.count_nonzero(union))

    return {
        "projected_long_side_px": projected_long,
        "projected_short_side_px": projected_short,
        "target_long_side_px": target_long,
        "target_short_side_px": target_short,
        "long_side_ratio": projected_long / target_long,
        "short_side_ratio": projected_short / target_short,
        "center_error_px": float(np.linalg.norm(projected_center - target_center)),
        "target_mask_area_px": target_area,
        "projected_mask_area_px": projected_area,
        "mask_intersection_area_px": intersection_area,
        "mask_union_area_px": union_area,
        "target_coverage": (
            intersection_area / target_area if target_area > 0 else 0.0
        ),
        "stl_precision": (
            intersection_area / projected_area if projected_area > 0 else 0.0
        ),
        "mask_iou": intersection_area / union_area if union_area > 0 else 0.0,
    }


def _draw_projected_box_diagnostic(
    ax,
    target_np,
    drr_display,
    projected_mask,
    projected_box,
    segmentation_diagnostic,
    box_metrics,
):
    """Draw final projected STL silhouette box vs target segmentation box."""
    ax.imshow(target_np, cmap="gray")
    if segmentation_diagnostic is not None:
        target_mask = np.asarray(segmentation_diagnostic["mask"], dtype=bool)
        if np.any(target_mask):
            target_overlay = np.ma.masked_where(
                ~target_mask, target_mask.astype(float)
            )
            ax.imshow(
                target_overlay,
                cmap="spring",
                alpha=0.25,
                vmin=0.0,
                vmax=1.0,
            )
    if projected_mask is not None and np.any(projected_mask):
        overlay = np.ma.masked_where(~projected_mask, projected_mask.astype(float))
        ax.imshow(overlay, cmap="cool", alpha=0.25, vmin=0.0, vmax=1.0)
    ax.imshow(drr_display, cmap="magma", alpha=0.25)

    legend_lines = []
    if segmentation_diagnostic is not None:
        _draw_box(
            ax,
            segmentation_diagnostic["box"],
            color="lime",
            linewidth=2.4,
            label="target",
        )
        legend_lines.append("green = segment target")
    if projected_box is not None:
        _draw_box(
            ax,
            projected_box,
            color="cyan",
            linewidth=2.4,
            linestyle="--",
            label="STL",
        )
        legend_lines.append("cyan = final STL")

    if box_metrics is not None:
        legend_lines.extend(
            [
                f"long ratio={box_metrics['long_side_ratio']:.2f}",
                f"short ratio={box_metrics['short_side_ratio']:.2f}",
                f"centre err={box_metrics['center_error_px']:.1f}px",
                f"target cover={100.0 * box_metrics['target_coverage']:.1f}%",
                f"STL in target={100.0 * box_metrics['stl_precision']:.1f}%",
                f"mask IoU={100.0 * box_metrics['mask_iou']:.1f}%",
            ]
        )

    if not legend_lines:
        legend_lines.append("box comparison unavailable")
    ax.text(
        0.02,
        0.98,
        "\n".join(legend_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
    )
    ax.set_title("Final STL mask/box vs target")
    ax.axis("off")


def _expanded_bbox_mask(shape, bbox, pad=6):
    """Return a mask covering bbox expanded by pad pixels."""
    height, width = shape
    x, y, box_width, box_height = [int(value) for value in bbox]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + box_width + pad)
    y1 = min(height, y + box_height + pad)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _edge_mask_from_binary(mask, thickness=1):
    """Return a visible edge mask from a binary region."""
    mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    if mask_uint8.max() == 0:
        return np.zeros_like(mask_uint8, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, kernel) > 0
    if thickness > 1:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1)
        )
        edge = cv2.dilate(edge.astype(np.uint8), dilate_kernel) > 0
    return edge


def _compute_edge_diagnostic(target_np, projected_mask, segmentation_diagnostic):
    """Compute X-ray target edges and final projected STL silhouette edges."""
    target_edge_source = "selected segmentation mask"
    if segmentation_diagnostic is not None:
        target_mask = np.asarray(segmentation_diagnostic["mask"], dtype=bool)
        target_edges = _edge_mask_from_binary(target_mask, thickness=1)
    else:
        # Fallback only when the selected segment_mode diagnostic is unavailable.
        target_edge_source = "fallback Otsu contour"
        target_uint8 = np.clip(target_np * 255, 0, 255).round().astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(target_uint8)
        _, target_mask = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        target_edges_uint8 = np.zeros_like(target_mask, dtype=np.uint8)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(target_edges_uint8, [contour], -1, 255, thickness=1)
            target_edges = target_edges_uint8 > 0
        else:
            target_edge_source = "fallback Canny edges"
            target_uint8 = np.clip(target_np * 255, 0, 255).round().astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(target_uint8)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            target_edges = cv2.Canny(blurred, 50, 150) > 0

    if not np.any(target_edges):
        # Last-resort fallback to intensity edges if the mask edge is empty.
        target_edge_source = "fallback Canny edges"
        target_uint8 = np.clip(target_np * 255, 0, 255).round().astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(target_uint8)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        target_edges = cv2.Canny(blurred, 50, 150) > 0
        if segmentation_diagnostic is not None:
            bbox_mask = _expanded_bbox_mask(
                target_edges.shape,
                segmentation_diagnostic["bbox_xywh"],
                pad=6,
            )
            target_edges &= bbox_mask

    stl_edges = _edge_mask_from_binary(projected_mask, thickness=1)

    tolerance_px = 2
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerance_px + 1, 2 * tolerance_px + 1),
    )
    target_dilated = cv2.dilate(target_edges.astype(np.uint8), kernel) > 0
    stl_dilated = cv2.dilate(stl_edges.astype(np.uint8), kernel) > 0

    target_count = int(np.count_nonzero(target_edges))
    stl_count = int(np.count_nonzero(stl_edges))
    target_matched = int(np.count_nonzero(target_edges & stl_dilated))
    stl_matched = int(np.count_nonzero(stl_edges & target_dilated))
    target_match_fraction = (
        target_matched / target_count if target_count > 0 else 0.0
    )
    stl_match_fraction = stl_matched / stl_count if stl_count > 0 else 0.0
    if target_match_fraction + stl_match_fraction > 0:
        f1 = (
            2.0
            * target_match_fraction
            * stl_match_fraction
            / (target_match_fraction + stl_match_fraction)
        )
    else:
        f1 = 0.0

    return {
        "target_edges": target_edges,
        "stl_edges": stl_edges,
        "metrics": {
            "tolerance_px": tolerance_px,
            "target_edge_count_px": target_count,
            "stl_edge_count_px": stl_count,
            "target_edge_source": target_edge_source,
            "target_match_fraction": float(target_match_fraction),
            "stl_match_fraction": float(stl_match_fraction),
            "edge_f1": float(f1),
        },
    }


def _draw_edge_overlay(ax, target_np, edge_diagnostic):
    """Draw target X-ray edges against final STL projected silhouette edges."""
    ax.imshow(target_np, cmap="gray")
    target_edges = edge_diagnostic["target_edges"]
    stl_edges = edge_diagnostic["stl_edges"]
    overlap = target_edges & stl_edges

    rgba = np.zeros((*target_edges.shape, 4), dtype=np.float32)
    rgba[target_edges] = [0.0, 1.0, 0.0, 0.95]   # green target edge
    rgba[stl_edges] = [0.0, 0.85, 1.0, 0.95]     # cyan STL edge
    rgba[overlap] = [1.0, 1.0, 1.0, 1.0]         # white exact overlap
    ax.imshow(rgba)

    metrics = edge_diagnostic["metrics"]
    target_label = (
        "segment target edge"
        if metrics.get("target_edge_source") == "selected segmentation mask"
        else metrics.get("target_edge_source", "target edge")
    )
    lines = [
        f"green = {target_label}",
        "cyan = projected STL edge",
        f"target match ±{metrics['tolerance_px']}px="
        f"{100.0 * metrics['target_match_fraction']:.1f}%",
        f"STL match ±{metrics['tolerance_px']}px="
        f"{100.0 * metrics['stl_match_fraction']:.1f}%",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
    )
    ax.set_title("Target edge vs STL edge")
    ax.axis("off")


def _draw_metrics_summary(
    ax,
    params,
    saved_pose_ncc,
    segmentation_diagnostic,
    box_metrics,
    edge_diagnostic,
    warm_start_history,
):
    """Draw compact numeric registration diagnostics."""
    rx_deg = np.degrees(params[0].item())
    ry_deg = np.degrees(params[1].item())
    rz_deg = np.degrees(params[2].item())
    tx = params[3].item()
    ty = params[4].item()
    tz = params[5].item()

    lines = [
        "Final diagnostics",
        f"NCC = {saved_pose_ncc:.4f}",
        f"rx/ry/rz = {rx_deg:+.2f}, {ry_deg:+.2f}, {rz_deg:+.2f} deg",
        f"tx/ty/tz = {tx:+.2f}, {ty:+.2f}, {tz:+.2f} mm",
    ]
    if segmentation_diagnostic is not None:
        area_pct = (
            100.0
            * segmentation_diagnostic["mask_area_px"]
            / max(segmentation_diagnostic["roi_area_px"], 1)
        )
        lines.append(
            f"segment = {segmentation_diagnostic['label']} ({area_pct:.1f}% ROI)"
        )
    if box_metrics is not None:
        lines.append(
            "box ratio long/short = "
            f"{box_metrics['long_side_ratio']:.2f}/"
            f"{box_metrics['short_side_ratio']:.2f}"
        )
        lines.append(
            "mask cover/IoU = "
            f"{100.0 * box_metrics['target_coverage']:.1f}%/"
            f"{100.0 * box_metrics['mask_iou']:.1f}%"
        )
    if edge_diagnostic is not None:
        metrics = edge_diagnostic["metrics"]
        lines.append(
            "edge match target/STL = "
            f"{100.0 * metrics['target_match_fraction']:.1f}%/"
            f"{100.0 * metrics['stl_match_fraction']:.1f}%"
        )
    if warm_start_history is not None:
        accepted = sum(1 for item in warm_start_history[1:] if item["accepted"])
        attempted = max(0, len(warm_start_history) - 1)
        final_delta = (
            warm_start_history[-1].get("ty_delta_mm")
            if len(warm_start_history) > 1
            else 0.0
        )
        lines.append(f"warm starts = {accepted}/{attempted} accepted")
        lines.append(f"last ty_delta = {final_delta:+.3f} mm")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=11,
        linespacing=1.35,
        bbox=dict(facecolor="black", alpha=0.72, edgecolor="none"),
    )
    ax.set_facecolor("0.08")
    ax.set_title("Numeric summary")
    ax.axis("off")


def _bbox_mismatch(projected_bbox, target_bbox, image_shape):
    """Compare oriented box size primarily, with a small centre penalty."""
    projected_sides = np.array(
        [projected_bbox["long_side_px"], projected_bbox["short_side_px"]],
        dtype=np.float64,
    )
    target_sides = np.array(
        [target_bbox["long_side_px"], target_bbox["short_side_px"]],
        dtype=np.float64,
    )
    size_error = float(np.abs(np.log(projected_sides / target_sides)).sum())

    projected_center = projected_bbox["center_xy"]
    target_center = target_bbox["center_xy"]
    center_error = np.linalg.norm(projected_center - target_center) / max(
        np.hypot(target_bbox["long_side_px"], target_bbox["short_side_px"]), 1.0
    )

    height, width = image_shape
    corners = np.asarray(projected_bbox["corners_xy"], dtype=np.float64)
    clipped = (
        corners[:, 0].min() < -0.5
        or corners[:, 1].min() < -0.5
        or corners[:, 0].max() > width - 0.5
        or corners[:, 1].max() > height - 0.5
    )
    clipping_penalty = 0.5 if clipped else 0.0
    return float(size_error + 0.20 * center_error + clipping_penalty)


def _projected_points_box(points_xy):
    """Return an oriented box for projected vertices, including off-image ones."""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("Projected STL points must have shape (N, 2)")
    points_xy = points_xy[np.isfinite(points_xy).all(axis=1)]
    if len(points_xy) < 3:
        return None

    contour = np.ascontiguousarray(points_xy.astype(np.float32)).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (box_width, box_height), _ = rect
    if box_width <= 1e-6 or box_height <= 1e-6:
        return None

    corners = cv2.boxPoints(rect).astype(np.float64)
    edge_vectors = np.roll(corners, -1, axis=0) - corners
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    long_edge = edge_vectors[int(np.argmax(edge_lengths))]
    angle_deg = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))
    x0 = int(np.floor(points_xy[:, 0].min()))
    x1 = int(np.ceil(points_xy[:, 0].max()))
    y0 = int(np.floor(points_xy[:, 1].min()))
    y1 = int(np.ceil(points_xy[:, 1].max()))

    return {
        "bbox_xywh": (x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
        "center_xy": np.array([center_x, center_y], dtype=np.float64),
        "corners_xy": corners,
        "long_side_px": float(max(box_width, box_height)),
        "short_side_px": float(min(box_width, box_height)),
        "angle_deg": angle_deg,
    }


def _load_centered_stl_vertices(stl_path, stl_scale, stl_center):
    """Load STL vertices in the centred world coordinates used by DiffDRR."""
    mesh = trimesh.load(stl_path, force="mesh")
    if stl_scale != 1.0:
        mesh.apply_scale(stl_scale)
    return (
        np.asarray(mesh.vertices, dtype=np.float64)
        - np.asarray(stl_center, dtype=np.float64)
    )


def _project_centered_vertices(vertices_world, params, renderer):
    """Project centred STL vertices without clipping them to the detector."""
    from diffdrr.pose import convert

    angles_zyx = torch.stack((params[2], params[1], params[0])).unsqueeze(0)
    translation = params[3:].unsqueeze(0)
    pose = convert(
        angles_zyx,
        translation,
        parameterization="euler_angles",
        convention="ZYX",
    )
    with torch.no_grad():
        source, detector_targets = renderer.detector(pose, calibration=None)
    source = source[0, 0].detach().cpu().numpy().astype(np.float64)
    detector_grid = (
        detector_targets[0]
        .reshape(renderer.detector.height, renderer.detector.width, 3)
        .detach().cpu().numpy().astype(np.float64)
    )

    detector_origin = detector_grid[0, 0]
    row_step = detector_grid[1, 0] - detector_origin
    column_step = detector_grid[0, 1] - detector_origin
    plane_normal = np.cross(row_step, column_step)
    rays = np.asarray(vertices_world, dtype=np.float64) - source
    denominator = rays @ plane_normal
    numerator = float(np.dot(detector_origin - source, plane_normal))
    valid = np.abs(denominator) > 1e-12
    ray_scale = np.zeros_like(denominator)
    ray_scale[valid] = numerator / denominator[valid]
    valid &= ray_scale > 0.0
    intersections = source + ray_scale[:, None] * rays
    offsets = intersections - detector_origin
    pixel_y = (offsets @ row_step) / np.dot(row_step, row_step)
    pixel_x = (offsets @ column_step) / np.dot(column_step, column_step)
    projected = np.column_stack((pixel_x, pixel_y))
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def _full_projected_stl_box(vertices_world, params, renderer):
    """Measure the complete STL projection, even beyond the ROI boundary."""
    projected, valid = _project_centered_vertices(
        vertices_world, params, renderer
    )
    return _projected_points_box(projected[valid])


def _depth_target_box(roi_img, segment_mode, implant_percentile):
    """Prefer a segmented target box, with an inset ROI fallback."""
    try:
        diagnostic = build_segmentation_diagnostic(
            roi_img, segment_mode, implant_percentile
        )
        return diagnostic["box"], diagnostic["label"], "segmentation"
    except Exception as error:
        height, width = roi_img.shape
        margin_x = int(round(0.05 * width)) if width >= 20 else 0
        margin_y = int(round(0.05 * height)) if height >= 20 else 0
        fallback_box = _box_from_bbox(
            (
                margin_x,
                margin_y,
                max(1, width - 2 * margin_x),
                max(1, height - 2 * margin_y),
            )
        )
        print(
            "[WARN] Target segmentation unavailable for depth fitting "
            f"({error}); fitting the STL silhouette inside the ROI instead"
        )
        return fallback_box, "ROI inset fallback", "roi_fallback"


def estimate_no_ruler_pixel_spacing(
    renderer,
    roi_img,
    base_params,
    device,
    pixel_spacing,
    stl_path,
    stl_scale,
    stl_center,
    implant_percentile=80.0,
    segment_mode="bone",
):
    """Infer an image-fit spacing so a nominal-depth silhouette fits its target.

    This resolves the visual scale/depth ambiguity when no physical ruler is
    available. The result is an effective fitting parameter, not a physical
    detector calibration.
    """
    target_box, target_label, target_source = _depth_target_box(
        roi_img, segment_mode, implant_percentile
    )
    vertices_world = _load_centered_stl_vertices(
        stl_path, stl_scale, stl_center
    )
    projected_box = _full_projected_stl_box(
        vertices_world, base_params, renderer
    )
    if projected_box is None:
        raise RuntimeError("Nominal-depth pose produced no projectable STL vertices")

    long_ratio = projected_box["long_side_px"] / max(
        target_box["long_side_px"], 1e-6
    )
    short_ratio = projected_box["short_side_px"] / max(
        target_box["short_side_px"], 1e-6
    )
    raw_factor = float(np.sqrt(long_ratio * short_ratio))
    if not np.isfinite(raw_factor) or raw_factor <= 0.0:
        raise RuntimeError(f"Invalid no-ruler image-fit scale factor {raw_factor}")
    scale_factor = float(np.clip(raw_factor, 0.25, 4.0))
    inferred_spacing = float(pixel_spacing) * scale_factor
    print(
        "[INFO] No-ruler image fit: effective pixel spacing "
        f"{float(pixel_spacing):.6f} -> {inferred_spacing:.6f} mm/pixel "
        f"(size factor={scale_factor:.4f}; not a physical calibration)"
    )
    return inferred_spacing, {
        "input_pixel_spacing_mm": float(pixel_spacing),
        "effective_pixel_spacing_mm": inferred_spacing,
        "scale_factor": scale_factor,
        "unclipped_scale_factor": raw_factor,
        "nominal_ty_mm": float(base_params[4].item()),
        "target_box_source": target_source,
        "target_box_label": target_label,
        "target_box": _serialise_box(target_box),
        "projected_box_before_rescale": _serialise_box(projected_box),
        "warning": (
            "Effective image-fit spacing only; no physical ruler calibration "
            "was available."
        ),
    }


def estimate_initial_ty(renderer, roi_img, base_params, device, sdd, out_dir,
                        steps=13, ty_min=None, ty_max=None,
                        implant_percentile=80.0, segment_mode="bone",
                        stl_path=None, stl_scale=1.0, stl_center=None):
    """Choose ty whose STL silhouette box best matches the bone/implant ROI box."""
    target_bbox, target_label, target_source = _depth_target_box(
        roi_img, segment_mode, implant_percentile
    )
    vertices_world = None
    if stl_path is not None and stl_center is not None:
        vertices_world = _load_centered_stl_vertices(
            stl_path, stl_scale, stl_center
        )

    if ty_min is None:
        ty_min = 0.65 * sdd
    if ty_max is None:
        ty_max = 0.97 * sdd
    if not 0.0 < ty_min < ty_max:
        raise ValueError("Automatic ty range must satisfy 0 < min < max")

    candidates = np.linspace(float(ty_min), float(ty_max), int(steps))
    scores = []
    projected_boxes = []
    silhouettes = []
    print(
        f"[INFO] Automatic depth calibration: sweeping {len(candidates)} ty "
        f"values from {ty_min:.1f} to {ty_max:.1f} mm"
    )

    with torch.no_grad():
        for candidate in candidates:
            params = base_params.detach().clone()
            params[4] = float(candidate)
            silhouette = render_drr(renderer, params, device)
            silhouette_np = silhouette.detach().cpu().numpy()
            maximum = float(np.max(silhouette_np))
            if vertices_world is not None:
                projected_bbox = _full_projected_stl_box(
                    vertices_world, params, renderer
                )
            else:
                projected_bbox = _mask_box(
                    silhouette_np > max(0.5 * maximum, 1e-6)
                )
            if projected_bbox is None:
                score = float("inf")
            else:
                score = _bbox_mismatch(
                    projected_bbox, target_bbox, roi_img.shape
                )
            scores.append(score)
            projected_boxes.append(projected_bbox)
            silhouettes.append(silhouette_np)
            print(
                f"  ty={candidate:8.2f} mm  bbox={projected_bbox}  "
                f"score={score:.5f}"
            )

    finite_scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(finite_scores).any():
        raise RuntimeError("No ty candidate produced a visible STL silhouette")
    best_index = int(np.nanargmin(finite_scores))
    best_ty = float(candidates[best_index])
    best_bbox = projected_boxes[best_index]
    best_silhouette = silhouettes[best_index]
    print(f"[INFO] Automatic initial ty selected: {best_ty:.3f} mm")

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].imshow(roi_img, cmap="gray")
    _draw_box(axes[0], target_bbox, color="lime", linewidth=2.2)
    axes[0].set_title(f"Depth target: {target_label}")
    axes[0].axis("off")

    axes[1].plot(candidates, scores, "o-", color="tab:blue")
    axes[1].axvline(best_ty, color="red", linestyle="--")
    axes[1].set_xlabel("ty (mm)")
    axes[1].set_ylabel("Bounding-box mismatch")
    axes[1].set_title(f"Depth Sweep (selected ty={best_ty:.1f} mm)")
    axes[1].grid(alpha=0.3)

    axes[2].imshow(roi_img, cmap="gray")
    visible = np.ma.masked_less_equal(best_silhouette, 0.05)
    axes[2].imshow(visible, cmap="magma", alpha=0.45)
    _draw_box(axes[2], best_bbox, color="cyan", linewidth=2.2)
    _draw_box(axes[2], target_bbox, color="lime", linewidth=2.2)
    axes[2].set_xlim(-0.5, roi_img.shape[1] - 0.5)
    axes[2].set_ylim(roi_img.shape[0] - 0.5, -0.5)
    axes[2].set_title(f"Full STL projection: cyan\n{target_label}: green")
    axes[2].axis("off")

    fig.tight_layout()
    figure_path = os.path.join(out_dir, "depth_calibration.png")
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    data = {
        "selected_ty_mm": best_ty,
        "target_box_source": target_source,
        "target_box_label": target_label,
        "target_box": _serialise_box(target_bbox),
        "selected_silhouette_box": _serialise_box(best_bbox),
        "projected_box_measurement": (
            "full_stl_vertices" if vertices_world is not None else "clipped_drr"
        ),
        "implant_threshold_percentile": float(implant_percentile),
        "candidates_ty_mm": candidates.tolist(),
        "bbox_mismatch_scores": [
            None if not np.isfinite(value) else float(value) for value in scores
        ],
    }
    json_path = os.path.join(out_dir, "depth_calibration.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved depth calibration: {figure_path}")
    print(f"[INFO] Saved depth calibration data: {json_path}")
    return best_ty


# ---------------------------------------------------------------------------
# 3.  DiffDRR renderer
# ---------------------------------------------------------------------------

def build_renderer(
    volume,
    affine,
    sdd,
    pixel_spacing,
    detector_height,
    detector_width,
    device,
    projection_mode="silhouette",
    silhouette_threshold=0.01,
    silhouette_blur_sigma=1.5,
):
    """Build a DiffDRR renderer from a torchio.Subject."""
    print("[INFO] Building DiffDRR renderer ...")

    # diffdrr.data.read adds the density, reorientation, mask, and centering
    # metadata required by DiffDRR >= 0.6.
    from diffdrr.data import read

    volume_image = tio.ScalarImage(
        tensor=volume.cpu(),
        affine=affine,
    )
    subject = read(
        volume_image,
        orientation="AP",
        center_volume=True,
    )

    from diffdrr.drr import DRR

    renderer = DRR(
        subject=subject,
        sdd=sdd,
        height=detector_height,
        delx=pixel_spacing,
        width=detector_width,
        dely=pixel_spacing,
        renderer="trilinear",
        reverse_x_axis=False,
    )
    renderer.to(device)
    renderer.projection_mode = projection_mode
    renderer.silhouette_threshold = silhouette_threshold
    renderer.silhouette_blur_sigma = silhouette_blur_sigma

    print(f"[INFO] Renderer ready. Detector: {detector_height}x{detector_width}, "
          f"SDD={sdd}mm, pixel={pixel_spacing}mm")
    print(f"[INFO] Projection mode: {projection_mode}")
    return renderer


# ---------------------------------------------------------------------------
# 4.  Pose utilities
# ---------------------------------------------------------------------------

def euler_to_matrix(angles_rad):
    """Convert ZYX Euler angles (3,) to rotation matrix (3, 3)."""
    rx, ry, rz = angles_rad
    zero = torch.zeros_like(rx)
    one = torch.ones_like(rx)
    Rx = torch.stack([
        one, zero, zero,
        zero, torch.cos(rx), -torch.sin(rx),
        zero, torch.sin(rx), torch.cos(rx),
    ]).reshape(3, 3)
    Ry = torch.stack([
        torch.cos(ry), zero, torch.sin(ry),
        zero, one, zero,
        -torch.sin(ry), zero, torch.cos(ry),
    ]).reshape(3, 3)
    Rz = torch.stack([
        torch.cos(rz), -torch.sin(rz), zero,
        torch.sin(rz), torch.cos(rz), zero,
        zero, zero, one,
    ]).reshape(3, 3)
    return Rz @ Ry @ Rx


def params_to_pose(params, device):
    """Convert 6-DOF params to rotation matrix + translation vector."""
    angles = params[:3]
    translation = params[3:]
    rotation = euler_to_matrix(angles).to(device)
    return rotation, translation


# ---------------------------------------------------------------------------
# 5.  NCC loss
# ---------------------------------------------------------------------------

def ncc_loss(pred, target):
    """Normalised Cross-Correlation loss (minimise 1 - NCC)."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()

    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    num = (pred_centered * target_centered).sum()
    denom = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum() + 1e-8)

    ncc = num / (denom + 1e-8)
    return 1.0 - ncc


def _sobel(image):
    """Differentiable Sobel magnitude filter for a 2D torch tensor."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    g = image.unsqueeze(0).unsqueeze(0)
    gx = torch.nn.functional.conv2d(g, kx, padding=1)
    gy = torch.nn.functional.conv2d(g, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8).squeeze(0).squeeze(0)


def edge_ncc_loss(pred, target, edge_weight=0.3):
    """Combined silhouette NCC + edge-gradient NCC.

    The edge term restores gradient signal along the X-rotation (beam) axis,
    where pure silhouette NCC has near-zero sensitivity because a filled
    silhouette's external outline barely changes under small beam-axis
    rotation.
    """
    base_loss = ncc_loss(pred, target)
    edge_loss = ncc_loss(_sobel(pred), _sobel(target))
    return (1.0 - edge_weight) * base_loss + edge_weight * edge_loss


# ---------------------------------------------------------------------------
# 6.  DRR rendering
# ---------------------------------------------------------------------------

def _gaussian_blur_2d(image, sigma):
    """Differentiable 2D Gaussian blur for a single-channel image."""
    if sigma <= 0:
        return image

    radius = max(1, int(np.ceil(3.0 * sigma)))
    coordinates = torch.arange(
        -radius, radius + 1, device=image.device, dtype=image.dtype
    )
    kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    blurred = torch.nn.functional.conv2d(
        image.unsqueeze(0).unsqueeze(0),
        kernel_2d,
        padding=radius,
    )
    return blurred.squeeze(0).squeeze(0)


def solid_external_silhouette(drr, threshold=0.01, blur_sigma=1.5):
    """Convert a line-integral DRR into a filled external silhouette.

    The forward pass uses OpenCV external contours, which removes internal
    attenuation detail and fills projected holes/cavities. A smooth occupancy
    mask supplies a straight-through gradient so pose optimisation remains
    differentiable despite the hard contour operation.
    """
    positive = torch.clamp_min(drr, 0.0)
    scale = positive.detach().amax().clamp_min(torch.finfo(drr.dtype).eps)
    normalised = positive / scale

    temperature = max(threshold * 0.5, 0.005)
    baseline = torch.sigmoid(
        torch.as_tensor(-threshold / temperature, device=drr.device, dtype=drr.dtype)
    )
    soft = (
        torch.sigmoid((normalised - threshold) / temperature) - baseline
    ) / (1.0 - baseline)
    soft = torch.clamp(soft, 0.0, 1.0)

    binary = (normalised.detach().cpu().numpy() > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(binary, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)

    hard = torch.from_numpy(filled).to(device=drr.device, dtype=drr.dtype)
    hard_blurred = _gaussian_blur_2d(hard, blur_sigma)
    soft_blurred = _gaussian_blur_2d(soft, blur_sigma)

    # Forward value = hard filled contour; backward derivative = soft mask.
    return hard_blurred + soft_blurred - soft_blurred.detach()


def render_drr(renderer, params, device, projection_mode=None):
    """Render a DRR from 6-DOF pose parameters.

    params: tensor [rx, ry, rz, tx, ty, tz]
    """
    # DiffDRR expects positional pose arguments. For a conventional XYZ
    # parameter vector whose rotation matrix is Rz @ Ry @ Rx, ZYX receives
    # the angles in [rz, ry, rx] order.
    angles = torch.stack((params[2], params[1], params[0])).unsqueeze(0)
    translation = params[3:].unsqueeze(0)   # (1, 3) translation in mm

    drr = renderer(
        angles,
        translation,
        parameterization="euler_angles",
        convention="ZYX",
    )

    # Squeeze to 2D
    if drr.dim() == 4:
        drr = drr.squeeze(0).squeeze(0)    # (H, W)
    elif drr.dim() == 3:
        drr = drr.squeeze(0)               # (H, W)

    active_projection = (
        getattr(renderer, "projection_mode", "silhouette")
        if projection_mode is None
        else projection_mode
    )
    if active_projection == "silhouette":
        drr = solid_external_silhouette(
            drr,
            threshold=renderer.silhouette_threshold,
            blur_sigma=renderer.silhouette_blur_sigma,
        )

    return drr


# ---------------------------------------------------------------------------
# 7.  Coarse grid search (optional)
# ---------------------------------------------------------------------------

def coarse_grid_search(renderer, target, init_params, device,
                       n_angles=5, angle_range_deg=30.0):
    """Brute-force search over rotation angles to find a good initialisation."""
    print("[INFO] Running coarse grid search for initialisation ...")
    best_loss = float("inf")
    best_params = None

    angles = np.linspace(-angle_range_deg, angle_range_deg, n_angles)
    total = n_angles ** 3
    count = 0
    base_params = init_params.detach()

    for rx_deg in angles:
        for ry_deg in angles:
            for rz_deg in angles:
                count += 1
                params = base_params.clone()
                params[:3] += torch.tensor(
                    [np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)],
                    dtype=torch.float32,
                    device=device,
                )
                params.requires_grad_(True)
                try:
                    drr = render_drr(renderer, params, device)
                    if drr.shape != target.shape:
                        drr = torch.nn.functional.interpolate(
                            drr.unsqueeze(0).unsqueeze(0),
                            size=target.shape,
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).squeeze(0)
                    loss = edge_ncc_loss(drr, target, 0.3)
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_params = params.detach().clone()
                        print(f"  [{count}/{total}] New best NCC={1 - best_loss:.4f} "
                              f"(rx={rx_deg:.1f} ry={ry_deg:.1f} rz={rz_deg:.1f})")
                except Exception:
                    continue

    if best_params is None:
        print("[WARN] Grid search failed, using default initialisation.")
        best_params = init_params.detach().clone().requires_grad_(True)
    else:
        print(f"[INFO] Grid search complete. Best NCC={1 - best_loss:.4f}")

    return best_params


# ---------------------------------------------------------------------------
# 8.  Optimisation
# ---------------------------------------------------------------------------

def optimize_pose(renderer, target, init_params, device,
                  coarse_iters=100, fine_iters=300, lr=5e-3,
                  translation_lr=0.5, coarse_scale=0.5, edge_weight=0.3,
                  lock_ty=False, ty_bounds=None):
    """Two-stage optimisation with unit-appropriate rotation/translation LRs."""
    rotation_params = init_params[:3].clone().detach().requires_grad_(True)
    translation_params = init_params[3:].clone().detach().requires_grad_(True)
    locked_ty = init_params[4].clone().detach()

    def enforce_ty_lock():
        with torch.no_grad():
            if lock_ty:
                translation_params[1].copy_(locked_ty)
            elif ty_bounds is not None:
                translation_params[1].clamp_(float(ty_bounds[0]), float(ty_bounds[1]))

    enforce_ty_lock()

    def current_params():
        return torch.cat((rotation_params, translation_params))

    def make_optimizer(rotation_lr, translation_step):
        return torch.optim.Adam([
            {"params": [rotation_params], "lr": rotation_lr},
            {"params": [translation_params], "lr": translation_step},
        ])

    # --- Coarse stage (downsampled) ---
    if coarse_iters > 0:
        coarse_h = max(1, int(target.shape[0] * coarse_scale))
        coarse_w = max(1, int(target.shape[1] * coarse_scale))
        target_coarse = torch.nn.functional.interpolate(
            target.unsqueeze(0).unsqueeze(0),
            size=(coarse_h, coarse_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        optimizer = make_optimizer(lr, translation_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=coarse_iters)

        print(f"\n[INFO] === Coarse stage: {coarse_iters} iters, "
              f"target {coarse_h}x{coarse_w}, rotation_lr={lr}, "
              f"translation_lr={translation_lr} ===")
        if lock_ty:
            print(f"[INFO] Warm-start ty lock active: ty={locked_ty.item():.3f} mm")
        elif ty_bounds is not None:
            print(
                f"[INFO] Calibrated ty guard active: "
                f"{ty_bounds[0]:.3f} <= ty <= {ty_bounds[1]:.3f} mm"
            )
        for it in range(coarse_iters):
            optimizer.zero_grad()
            params = current_params()
            try:
                drr = render_drr(renderer, params, device)
            except Exception as e:
                print(f"[WARN] Render failed at iter {it}: {e}")
                continue

            if drr.shape != target_coarse.shape:
                drr = torch.nn.functional.interpolate(
                    drr.unsqueeze(0).unsqueeze(0),
                    size=target_coarse.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            loss = edge_ncc_loss(drr, target_coarse, edge_weight)
            loss.backward()
            optimizer.step()
            enforce_ty_lock()
            scheduler.step()

            if it % 10 == 0 or it == coarse_iters - 1:
                ncc = 1 - loss.item()
                rotation_step, translation_step = scheduler.get_last_lr()
                print(f"  [coarse {it:4d}/{coarse_iters}] NCC={ncc:.6f}  loss={loss.item():.6f}  "
                      f"rotation_lr={rotation_step:.6f}  "
                      f"translation_lr={translation_step:.6f}")

    # --- Fine stage (full resolution) ---
    if fine_iters > 0:
        fine_lr = lr * 0.1
        fine_translation_lr = translation_lr * 0.1
        optimizer = make_optimizer(fine_lr, fine_translation_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_iters)

        print(f"\n[INFO] === Fine stage: {fine_iters} iters, "
              f"target {target.shape[0]}x{target.shape[1]}, "
              f"rotation_lr={fine_lr}, translation_lr={fine_translation_lr} ===")
        best_ncc = -float("inf")
        best_params = current_params().detach().clone()

        for it in range(fine_iters):
            optimizer.zero_grad()
            params = current_params()
            try:
                drr = render_drr(renderer, params, device)
            except Exception as e:
                print(f"[WARN] Render failed at iter {it}: {e}")
                continue

            if drr.shape != target.shape:
                drr = torch.nn.functional.interpolate(
                    drr.unsqueeze(0).unsqueeze(0),
                    size=target.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            loss = edge_ncc_loss(drr, target, edge_weight)
            loss.backward()
            optimizer.step()
            enforce_ty_lock()
            scheduler.step()

            ncc_val = 1 - loss.item()
            if ncc_val > best_ncc:
                best_ncc = ncc_val
                best_params = current_params().detach().clone()

            if it % 50 == 0 or it == fine_iters - 1:
                rotation_step, translation_step = scheduler.get_last_lr()
                print(f"  [fine {it:4d}/{fine_iters}] NCC={ncc_val:.6f}  loss={loss.item():.6f}  "
                      f"best={best_ncc:.6f}  rotation_lr={rotation_step:.6f}  "
                      f"translation_lr={translation_step:.6f}")

        print(f"\n[INFO] Best NCC (fine): {best_ncc:.6f}")
        return best_params, best_ncc

    params = current_params().detach()
    with torch.no_grad():
        drr = render_drr(renderer, params, device)
        if drr.shape != target.shape:
            drr = torch.nn.functional.interpolate(
                drr.unsqueeze(0).unsqueeze(0),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        final_ncc = 1.0 - ncc_loss(drr, target).item()
    return params, final_ncc


def evaluate_pose_ncc(renderer, target, params, device, projection_mode=None):
    """Render a pose and return its NCC against the target image."""
    with torch.no_grad():
        drr = render_drr(
            renderer, params.detach(), device, projection_mode=projection_mode
        )
        if drr.shape != target.shape:
            drr = torch.nn.functional.interpolate(
                drr.unsqueeze(0).unsqueeze(0),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        return 1.0 - ncc_loss(drr, target).item()


def _print_progress_bar(prefix, current, total, start_time, width=28):
    """Render an in-place terminal progress bar with elapsed time and ETA."""
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    fraction = current / total
    filled = int(round(width * fraction))
    elapsed = max(0.0, time.perf_counter() - start_time)
    bar = "#" * filled + "-" * (width - filled)
    message = (
        f"\r{prefix} |{bar}| {current:>4d}/{total:<4d} "
        f"({100.0 * fraction:5.1f}%) elapsed {elapsed:6.1f}s"
    )
    if 0 < current < total:
        eta = elapsed * (total - current) / current
        message += f" eta {eta:6.1f}s"
    sys.stdout.write(message)
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _add_overlay_text(ax, lines, x=0.02, y=0.98):
    """Add a compact annotation box to the current axes."""
    if not lines:
        return
    ax.text(
        x,
        y,
        "\n".join(str(line) for line in lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
        zorder=9,
    )


# ---------------------------------------------------------------------------
# 8b.  Post-optimisation refinement (rx polish + Chamfer)
# ---------------------------------------------------------------------------

def rx_polish_sweep(renderer, target, params, device,
                    range_deg=6.0, step_deg=0.25,
                    projection_mode="silhouette"):
    """1-D sweep over rx after gradient descent to recover the weak-gradient axis.

    A filled silhouette is nearly invariant to small X-rotations, so the NCC
    gradient w.r.t. rx is approximately zero and Adam cannot converge along it.
    This brute-force sweep sidesteps the dead gradient entirely.
    """
    base = params.detach().clone()
    best_ncc = evaluate_pose_ncc(
        renderer, target, base, device, projection_mode=projection_mode
    )
    best_params = base.clone()
    rxs = np.arange(-range_deg, range_deg + step_deg, step_deg)
    print(
        f"[INFO] rx polish sweep: {len(rxs)} candidates "
        f"over [{rxs[0]:.2f}, {rxs[-1]:.2f}] deg"
    )
    with torch.no_grad():
        for rx_deg in rxs:
            p = base.clone()
            p[0] = p[0] + np.radians(rx_deg)
            drr = render_drr(renderer, p, device, projection_mode=projection_mode)
            if drr.shape != target.shape:
                drr = torch.nn.functional.interpolate(
                    drr.unsqueeze(0).unsqueeze(0),
                    size=target.shape, mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)
            ncc = 1.0 - ncc_loss(drr, target).item()
            if ncc > best_ncc:
                best_ncc = ncc
                best_params = p.clone()
    delta = np.degrees(best_params[0].item() - base[0].item())
    print(f"[INFO] rx polish: best d_rx={delta:+.3f} deg, NCC={best_ncc:.6f}")
    return best_params, best_ncc


def chamfer_refinement(renderer, target, params, device, out_dir,
                       ty_range_mm=30.0, ty_step_mm=2.0,
                       rx_range_deg=4.0, rx_step_deg=0.25,
                       pixel_spacing=0.194, save_diagnostic=True,
                       ty_bounds=None, target_mask=None,
                       target_segment_mode="implant",
                       implant_percentile=80.0,
                       combined_objective=False,
                       iou_weight=1.0, outside_weight=1.0):
    """2-D grid refinement over (ty, rx) using a selected target contour.

    ``target_segment_mode`` independently selects an implant percentile mask or
    an Otsu bone mask. When ``combined_objective`` is enabled, candidate poses
    are ranked by normalized bidirectional Chamfer distance + mask-IoU loss +
    the fraction of projected STL pixels outside the filled target contour.
    """
    from scipy.spatial import cKDTree

    if target_segment_mode not in ("implant", "bone"):
        raise ValueError(
            "target_segment_mode must be either 'implant' or 'bone'"
        )
    if iou_weight < 0.0 or outside_weight < 0.0:
        raise ValueError("Contour-objective weights must be non-negative")

    # --- Extract the selected external target contour ---
    target_np = target.detach().cpu().numpy()
    if target_mask is None:
        target_uint8 = np.clip(target_np * 255, 0, 255).astype(np.uint8)
        try:
            if target_segment_mode == "bone":
                _, target_mask = segment_bone_bbox(target_uint8)
            else:
                _, target_mask = segment_implant_bbox(
                    target_uint8, percentile=implant_percentile
                )
        except Exception as error:
            print(
                "[WARN] Chamfer refinement: could not build "
                f"{target_segment_mode} target ({error}), skipping."
            )
            return params, 0.0

    target_mask = np.ascontiguousarray(
        np.asarray(target_mask, dtype=np.uint8) > 0,
        dtype=np.uint8,
    )
    target_contour = _largest_mask_contour(target_mask)
    if target_contour is None:
        print(
            f"[WARN] Chamfer refinement: no {target_segment_mode} "
            "contour found, skipping."
        )
        return params, 0.0

    target_contour = target_contour.squeeze(1).astype(np.float64)
    if len(target_contour) < 3:
        print(
            f"[WARN] Chamfer refinement: {target_segment_mode} contour "
            "too small, skipping."
        )
        return params, 0.0

    # Fill only the largest external contour. Internal threshold holes should
    # not be interpreted as implant/bone boundaries by IoU or outside penalty.
    filled_target_mask = np.zeros_like(target_mask, dtype=np.uint8)
    cv2.drawContours(
        filled_target_mask,
        [target_contour.astype(np.int32).reshape(-1, 1, 2)],
        -1,
        color=1,
        thickness=cv2.FILLED,
    )
    filled_target_bool = filled_target_mask.astype(bool)
    target_tree = cKDTree(target_contour)
    image_diagonal = max(float(np.hypot(*target_mask.shape)), 1.0)
    objective_label = (
        "combined contour" if combined_objective else "bidirectional Chamfer"
    )
    print(
        f"[INFO] Chamfer target: mode={target_segment_mode}, "
        f"objective={objective_label}"
    )

    # --- Grid search ---
    base = params.detach().clone()
    base_ty = base[4].item()
    base_rx = base[0].item()

    tys = np.arange(
        base_ty - ty_range_mm, base_ty + ty_range_mm + ty_step_mm, ty_step_mm
    )
    if ty_bounds is not None:
        ty_min, ty_max = float(ty_bounds[0]), float(ty_bounds[1])
        tys = tys[(tys >= ty_min) & (tys <= ty_max)]
        if len(tys) == 0:
            tys = np.array([float(np.clip(base_ty, ty_min, ty_max))])
    rxs = np.arange(
        -rx_range_deg, rx_range_deg + rx_step_deg, rx_step_deg
    )

    best_objective = float("inf")
    best_chamfer = float("inf")
    best_iou = 0.0
    best_outside_fraction = 1.0
    best_params = base.clone()

    print(
        f"[INFO] Chamfer refinement: {len(tys)}x{len(rxs)} grid "
        f"over ty=[{tys[0]:.1f},{tys[-1]:.1f}]mm "
        f"rx=[{rxs[0]:.2f},{rxs[-1]:.2f}]deg"
    )
    total_candidates = int(len(tys) * len(rxs))
    progress_count = 0
    progress_update_every = max(1, total_candidates // 100)
    progress_start_time = time.perf_counter()
    _print_progress_bar(
        "[INFO] Chamfer refinement progress:",
        progress_count,
        total_candidates,
        progress_start_time,
    )

    with torch.no_grad():
        for ty_val in tys:
            for rx_deg in rxs:
                progress_count += 1
                if (
                    progress_count == 1
                    or progress_count == total_candidates
                    or progress_count % progress_update_every == 0
                ):
                    _print_progress_bar(
                        "[INFO] Chamfer refinement progress:",
                        progress_count,
                        total_candidates,
                        progress_start_time,
                    )
                p = base.clone()
                p[4] = float(ty_val)
                p[0] = base_rx + np.radians(rx_deg)

                drr = render_drr(renderer, p, device)
                if drr.shape != target.shape:
                    drr = torch.nn.functional.interpolate(
                        drr.unsqueeze(0).unsqueeze(0),
                        size=target.shape, mode="bilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)

                drr_np = drr.detach().cpu().numpy()
                drr_uint8 = (np.clip(drr_np, 0, 1) * 255).astype(np.uint8)
                _, drr_mask = cv2.threshold(
                    drr_uint8, 30, 255, cv2.THRESH_BINARY
                )
                drr_contours, _ = cv2.findContours(
                    drr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not drr_contours:
                    continue
                drr_contour = max(drr_contours, key=cv2.contourArea)
                drr_contour = drr_contour.squeeze(1).astype(np.float64)
                if len(drr_contour) < 3:
                    continue

                drr_tree = cKDTree(drr_contour)
                d_to_target, _ = target_tree.query(drr_contour)
                target_to_drr, _ = drr_tree.query(target_contour)
                chamfer = float(
                    d_to_target.mean() + target_to_drr.mean()
                )

                projected_bool = drr_mask.astype(bool)
                intersection = int(
                    np.count_nonzero(projected_bool & filled_target_bool)
                )
                union = int(
                    np.count_nonzero(projected_bool | filled_target_bool)
                )
                projected_area = int(np.count_nonzero(projected_bool))
                iou = intersection / union if union > 0 else 0.0
                outside_fraction = (
                    np.count_nonzero(projected_bool & ~filled_target_bool)
                    / projected_area
                    if projected_area > 0
                    else 1.0
                )
                if combined_objective:
                    objective = (
                        chamfer / image_diagonal
                        + iou_weight * (1.0 - iou)
                        + outside_weight * outside_fraction
                    )
                else:
                    objective = chamfer

                if objective < best_objective:
                    best_objective = float(objective)
                    best_chamfer = chamfer
                    best_iou = float(iou)
                    best_outside_fraction = float(outside_fraction)
                    best_params = p.clone()

    if not np.isfinite(best_objective):
        print(
            "[WARN] Chamfer refinement: no candidate produced a valid "
            "projected contour; keeping the input pose."
        )
        return base, float("inf")

    delta_ty = best_params[4].item() - base_ty
    delta_rx = np.degrees(best_params[0].item() - base_rx)
    summary = (
        f"[INFO] Chamfer refinement: best d_ty={delta_ty:+.2f}mm "
        f"d_rx={delta_rx:+.3f}deg chamfer={best_chamfer:.2f}px"
    )
    if combined_objective:
        summary += (
            f" IoU={best_iou:.4f} outside={best_outside_fraction:.4f} "
            f"objective={best_objective:.6f}"
        )
    print(summary)

    if save_diagnostic:
        # --- Save Chamfer diagnostic figure ---
        os.makedirs(out_dir, exist_ok=True)
        with torch.no_grad():
            best_drr = render_drr(renderer, best_params, device)
            if best_drr.shape != target.shape:
                best_drr = torch.nn.functional.interpolate(
                    best_drr.unsqueeze(0).unsqueeze(0),
                    size=target.shape, mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)
            best_drr_np = best_drr.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(target_np, cmap="gray")
        tc = target_contour.astype(int)
        axes[0].plot(tc[:, 0], tc[:, 1], "lime", linewidth=1.5)
        axes[0].set_title(
            f"X-ray ROI + {target_segment_mode} contour"
        )
        axes[0].axis("off")

        axes[1].imshow(best_drr_np, cmap="gray")
        if combined_objective:
            axes[1].set_title(
                f"Best DRR (objective={best_objective:.3f})\n"
                f"Chamfer={best_chamfer:.1f}px, IoU={best_iou:.3f}, "
                f"outside={best_outside_fraction:.3f}"
            )
        else:
            axes[1].set_title(
                f"Best DRR (chamfer={best_chamfer:.1f}px)"
            )
        axes[1].axis("off")

        axes[2].imshow(target_np, cmap="gray")
        axes[2].imshow(best_drr_np, cmap="magma", alpha=0.40)
        axes[2].set_title(
            f"Overlay after {objective_label} refinement"
        )
        axes[2].axis("off")

        fig.tight_layout()
        fig_path = os.path.join(out_dir, "chamfer_refinement.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved Chamfer refinement diagnostic: {fig_path}")

    return best_params, best_chamfer


def run_refinement_pipeline(renderer, target, init_params, device, args,
                            pixel_spacing, save_chamfer_diagnostic=False,
                            lock_ty=False, ty_bounds=None,
                            chamfer_segmentation=None):
    """Run the optimisation/refinement stages once from a given initial pose."""
    best_params, optimized_ncc = optimize_pose(
        renderer, target, init_params, device,
        coarse_iters=args.coarse_iters,
        fine_iters=args.fine_iters,
        lr=args.lr,
        translation_lr=args.translation_lr,
        edge_weight=args.edge_ncc_weight,
        lock_ty=lock_ty,
        ty_bounds=ty_bounds,
    )

    best_params, optimized_ncc = rx_polish_sweep(
        renderer, target, best_params, device,
        range_deg=args.rx_polish_range,
        step_deg=args.rx_polish_step,
    )

    chamfer_dist = 0.0
    if args.chamfer_refine:
        chamfer_ty_range = 0.0 if lock_ty else args.chamfer_ty_range
        best_params, chamfer_dist = chamfer_refinement(
            renderer, target, best_params, device, args.out_dir,
            ty_range_mm=chamfer_ty_range,
            ty_step_mm=args.chamfer_ty_step,
            rx_range_deg=args.chamfer_rx_range,
            rx_step_deg=args.chamfer_rx_step,
            pixel_spacing=pixel_spacing,
            save_diagnostic=save_chamfer_diagnostic,
            ty_bounds=ty_bounds,
            target_mask=(
                None
                if chamfer_segmentation is None
                else chamfer_segmentation["mask"]
            ),
            target_segment_mode=args.chamfer_segment_mode,
            implant_percentile=args.implant_percentile,
            combined_objective=args.chamfer_combined_objective,
            iou_weight=args.chamfer_iou_weight,
            outside_weight=args.chamfer_outside_weight,
        )
    if lock_ty:
        best_params[4] = init_params[4].detach()

    optimized_ncc = evaluate_pose_ncc(renderer, target, best_params, device)
    return best_params.detach().clone(), float(optimized_ncc), float(chamfer_dist)


def run_warm_start_pipeline(renderer, target, init_params, device, args,
                            pixel_spacing, stl_path, stl_scale, stl_center,
                            chamfer_segmentation=None):
    """Optionally recycle the current best pose through extra refinement passes."""
    ty_bounds = None
    if args.ty_guard_mm > 0.0:
        anchor_ty = float(init_params[4].item())
        ty_bounds = (anchor_ty - args.ty_guard_mm, anchor_ty + args.ty_guard_mm)

    best_params, best_ncc, chamfer_dist = run_refinement_pipeline(
        renderer,
        target,
        init_params,
        device,
        args,
        pixel_spacing,
        save_chamfer_diagnostic=False,
        ty_bounds=ty_bounds,
        chamfer_segmentation=chamfer_segmentation,
    )
    history = [
        {
            "pass_index": 0,
            "kind": "initial",
            "ncc": float(best_ncc),
            "ty_mm": float(best_params[4].item()),
            "accepted": True,
            "improvement": None,
        }
    ]

    for pass_index in range(1, int(args.warm_start_passes) + 1):
        print(
            f"[INFO] Warm start pass {pass_index}/{args.warm_start_passes}: "
            f"restarting from NCC={best_ncc:.6f}"
        )
        candidate_params, candidate_ncc, candidate_chamfer = run_refinement_pipeline(
            renderer,
            target,
            best_params.detach().clone(),
            device,
            args,
            pixel_spacing,
            save_chamfer_diagnostic=False,
            lock_ty=not args.warm_start_free_ty,
            ty_bounds=ty_bounds,
            chamfer_segmentation=chamfer_segmentation,
        )
        improvement = float(candidate_ncc - best_ncc)
        ty_delta = float(candidate_params[4].item() - best_params[4].item())
        accepted = improvement > args.warm_start_min_delta
        history.append(
            {
                "pass_index": int(pass_index),
                "kind": "warm_start",
                "ncc": float(candidate_ncc),
                "ty_mm": float(candidate_params[4].item()),
                "ty_delta_mm": ty_delta,
                "accepted": bool(accepted),
                "improvement": improvement,
            }
        )
        show_warm_start_diagnostic(
            renderer,
            target,
            candidate_params,
            device,
            stl_path,
            stl_scale,
            stl_center,
            title=(
                f"Warm Start Pass {pass_index}\n"
                "STL detail diagnostic attenuation + full wireframe"
            ),
            overlay_lines=[
                f"pass: {pass_index}/{args.warm_start_passes}",
                f"ty: {candidate_params[4].item():.3f} mm",
                f"d_ty: {ty_delta:+.3f} mm",
                f"NCC: {candidate_ncc:.6f}",
                f"delta NCC: {improvement:+.6f}",
                "ty mode: free" if args.warm_start_free_ty else "ty mode: locked",
                "status: accepted" if accepted else "status: rejected",
            ],
        )
        if accepted:
            print(
                f"[INFO] Warm start pass {pass_index} accepted: "
                f"NCC improved by {improvement:+.6f} to {candidate_ncc:.6f}"
            )
            best_params = candidate_params
            best_ncc = candidate_ncc
            chamfer_dist = candidate_chamfer
        else:
            print(
                f"[INFO] Warm start pass {pass_index} rejected: "
                f"improvement {improvement:+.6f} <= "
                f"threshold {args.warm_start_min_delta:.6f}"
            )
            break

    if args.chamfer_refine:
        best_params, chamfer_dist = chamfer_refinement(
            renderer,
            target,
            best_params,
            device,
            args.out_dir,
            ty_range_mm=0.0,
            ty_step_mm=max(args.chamfer_ty_step, 1.0),
            rx_range_deg=0.0,
            rx_step_deg=max(args.chamfer_rx_step, 0.25),
            pixel_spacing=pixel_spacing,
            save_diagnostic=True,
            ty_bounds=ty_bounds,
            target_mask=(
                None
                if chamfer_segmentation is None
                else chamfer_segmentation["mask"]
            ),
            target_segment_mode=args.chamfer_segment_mode,
            implant_percentile=args.implant_percentile,
            combined_objective=args.chamfer_combined_objective,
            iou_weight=args.chamfer_iou_weight,
            outside_weight=args.chamfer_outside_weight,
        )
        best_ncc = evaluate_pose_ncc(renderer, target, best_params, device)

    return best_params.detach().clone(), float(best_ncc), float(chamfer_dist), history


# ---------------------------------------------------------------------------
# 9.  Output
# ---------------------------------------------------------------------------

def _normalise_for_display(image):
    """Robustly map a floating-point image to [0, 1] for saved figures."""
    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    lo, hi = np.percentile(image, (1.0, 99.0))
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def preview_initial_alignment(renderer, target, init_params, device, out_dir,
                              stl_path, stl_scale, stl_center,
                              show_window=True, ask_to_continue=True):
    """Save and optionally display the initial STL silhouette over the ROI."""
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        initial_drr = render_drr(renderer, init_params, device)
        initial_detail_drr = render_drr(
            renderer, init_params, device, projection_mode="attenuation"
        )
        if initial_drr.shape != target.shape:
            initial_drr = torch.nn.functional.interpolate(
                initial_drr.unsqueeze(0).unsqueeze(0),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        if initial_detail_drr.shape != target.shape:
            initial_detail_drr = torch.nn.functional.interpolate(
                initial_detail_drr.unsqueeze(0).unsqueeze(0),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        initial_ncc = 1.0 - ncc_loss(initial_drr, target).item()

    target_np = target.detach().cpu().numpy()
    drr_np = _normalise_for_display(initial_drr.detach().cpu().numpy())
    detail_np = _normalise_for_display(
        initial_detail_drr.detach().cpu().numpy()
    )
    wire_segments = project_stl_wireframe(
        stl_path, stl_scale, init_params, renderer, stl_center
    )
    axis_segments = project_stl_axes(
        stl_path, stl_scale, init_params, renderer
    )

    fig, axes = plt.subplots(1, 4, figsize=(21, 6))
    axes[0].imshow(target_np, cmap="gray")
    axes[0].set_title("Radiograph ROI")
    axes[0].axis("off")

    axes[1].imshow(drr_np, cmap="gray")
    axes[1].set_title("Initial STL Silhouette")
    axes[1].axis("off")

    axes[2].imshow(target_np, cmap="gray")
    axes[2].imshow(drr_np, cmap="magma", alpha=0.45)
    axes[2].set_title(f"Initial Overlay (NCC={initial_ncc:.4f})")
    axes[2].axis("off")

    _add_stl_diagnostic_overlay(
        axes[3],
        target_np,
        detail_np,
        wire_segments,
        axis_segments=axis_segments,
        title="Initial STL Spatial Position\nattenuation + full wireframe",
    )

    if show_window and ask_to_continue:
        fig.subplots_adjust(bottom=0.17)
        fig.suptitle(
            "Check the initial STL position. Continue only if this starting point is OK.",
            fontsize=13,
        )
    else:
        plt.tight_layout()
    preview_path = os.path.join(out_dir, "initial_alignment.png")
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved initial alignment preview: {preview_path}")

    continue_optimisation = True
    if show_window:
        backend = str(matplotlib.get_backend()).lower()
        if "agg" in backend:
            print("[WARN] The active Matplotlib backend cannot open a window.")
            print(f"       Inspect {preview_path} to check the initial alignment.")
        else:
            decision = {"continue": True}
            buttons = []
            if ask_to_continue:
                continue_ax = fig.add_axes([0.39, 0.035, 0.10, 0.055])
                stop_ax = fig.add_axes([0.51, 0.035, 0.10, 0.055])
                continue_button = Button(continue_ax, "Continue")
                stop_button = Button(stop_ax, "Stop")

                def accept(_event=None):
                    decision["continue"] = True
                    plt.close(fig)

                def reject(_event=None):
                    decision["continue"] = False
                    plt.close(fig)

                def on_key(event):
                    if event.key in ("enter", "return"):
                        accept(event)
                    elif event.key in ("escape", "q", "Q", "n", "N"):
                        reject(event)

                continue_button.on_clicked(accept)
                stop_button.on_clicked(reject)
                fig.canvas.mpl_connect("key_press_event", on_key)
                buttons.extend([continue_button, stop_button])
                print(
                    "[INFO] Inspect the initial overlay, then click Continue "
                    "or Stop."
                )
            else:
                print("[INFO] Inspect the initial overlay, then close the window.")
            plt.show()
            if ask_to_continue:
                continue_optimisation = bool(decision["continue"])

    plt.close(fig)
    return continue_optimisation


def show_warm_start_diagnostic(renderer, target, params, device,
                               stl_path, stl_scale, stl_center,
                               title, overlay_lines):
    """Show a non-saved warm-start diagnostic window with pose metadata."""
    backend = str(matplotlib.get_backend()).lower()
    if "agg" in backend:
        print("[WARN] Warm-start diagnostic window skipped: Agg backend is active.")
        return

    with torch.no_grad():
        detail_drr = render_drr(
            renderer, params, device, projection_mode="attenuation"
        )
        if detail_drr.shape != target.shape:
            detail_drr = torch.nn.functional.interpolate(
                detail_drr.unsqueeze(0).unsqueeze(0),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

    target_np = target.detach().cpu().numpy()
    detail_display = _normalise_for_display(detail_drr.detach().cpu().numpy())
    wire_segments = project_stl_wireframe(
        stl_path, stl_scale, params, renderer, stl_center
    )
    axis_segments = project_stl_axes(
        stl_path, stl_scale, params, renderer
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _add_stl_diagnostic_overlay(
        ax,
        target_np,
        detail_display,
        wire_segments,
        axis_segments=axis_segments,
        title=title,
    )
    _add_overlay_text(ax, overlay_lines)
    fig.tight_layout()
    print("[INFO] Inspect the warm-start diagnostic, then close the window.")
    plt.show()
    plt.close(fig)


def compute_pose_outputs(params, renderer, stl_center):
    """Return model transforms with explicit coordinate direction.

    DiffDRR optimises a camera pose around a centred STL volume. For reporting,
    also expose a screen-centred frame whose origin is the middle of the
    detector/image plane, so the translation does not depend on the STL file's
    original coordinate origin.
    """
    from diffdrr.pose import convert
    from scipy.spatial.transform import Rotation as SciPyRotation

    angles_zyx = torch.stack((params[2], params[1], params[0])).unsqueeze(0)
    translation = params[3:].unsqueeze(0)
    parameter_pose = convert(
        angles_zyx,
        translation,
        parameterization="euler_angles",
        convention="ZYX",
    ).matrix[0]

    # DiffDRR moves the source/detector around a fixed, centred volume.
    # Its complete camera-to-world pose also includes the AP reorientation.
    reorient = renderer.detector.reorient.matrix[0].to(parameter_pose)
    camera_to_centered_stl = parameter_pose @ reorient

    # Map coordinates from the original STL frame into the centred world used
    # during rendering, then invert the camera pose to obtain STL -> camera.
    stl_to_centered = torch.eye(4, device=params.device, dtype=params.dtype)
    stl_to_centered[:3, 3] = -torch.as_tensor(
        stl_center, device=params.device, dtype=params.dtype
    )
    stl_to_camera = torch.linalg.inv(camera_to_centered_stl) @ stl_to_centered

    with torch.no_grad():
        _source, detector_targets = renderer.detector(
            convert(
                angles_zyx,
                translation,
                parameterization="euler_angles",
                convention="ZYX",
            ),
            calibration=None,
        )
    detector_grid = detector_targets[0].reshape(
        renderer.detector.height,
        renderer.detector.width,
        3,
    )
    detector_center_world = detector_grid.reshape(-1, 3).mean(dim=0)
    world_to_camera = torch.linalg.inv(camera_to_centered_stl)
    detector_center_h = torch.cat(
        (
            detector_center_world.to(device=params.device, dtype=params.dtype),
            torch.ones(1, device=params.device, dtype=params.dtype),
        )
    )
    detector_center_camera = (world_to_camera @ detector_center_h)[:3]

    camera_to_screen_centered = torch.eye(
        4, device=params.device, dtype=params.dtype
    )
    camera_to_screen_centered[:3, 3] = -detector_center_camera
    centered_stl_to_screen_centered = camera_to_screen_centered @ world_to_camera
    stl_to_screen_centered = camera_to_screen_centered @ stl_to_camera

    centered_stl_to_screen_centered_np = (
        centered_stl_to_screen_centered.detach().cpu().numpy()
    )
    stl_to_camera_np = stl_to_camera.detach().cpu().numpy()
    stl_to_screen_centered_np = stl_to_screen_centered.detach().cpu().numpy()
    detector_center_camera_np = detector_center_camera.detach().cpu().numpy()
    model_xyz_deg = SciPyRotation.from_matrix(
        stl_to_camera_np[:3, :3]
    ).as_euler("xyz", degrees=True)

    return (
        camera_to_centered_stl.detach().cpu().numpy(),
        stl_to_camera_np,
        centered_stl_to_screen_centered_np,
        stl_to_screen_centered_np,
        detector_center_camera_np,
        model_xyz_deg,
    )


def project_stl_wireframe(stl_path, stl_scale, params, renderer, stl_center):
    """Project STL edges using DiffDRR's exact source and detector geometry."""
    from diffdrr.pose import convert

    mesh = trimesh.load(stl_path, force="mesh")
    if stl_scale != 1.0:
        mesh.apply_scale(stl_scale)

    # DiffDRR canonicalizes the volume by moving this STL centre to world zero.
    vertices_world = (
        np.asarray(mesh.vertices, dtype=np.float64)
        - np.asarray(stl_center, dtype=np.float64)
    )

    angles_zyx = torch.stack((params[2], params[1], params[0])).unsqueeze(0)
    translation = params[3:].unsqueeze(0)
    pose = convert(
        angles_zyx,
        translation,
        parameterization="euler_angles",
        convention="ZYX",
    )

    # These are the exact world-space source and detector-pixel centres used
    # by renderer.forward for this pose, including AP reorientation, axis
    # reversal, detector spacing, SDD, and principal-point calibration.
    with torch.no_grad():
        source, detector_targets = renderer.detector(pose, calibration=None)
    source = source[0, 0].detach().cpu().numpy().astype(np.float64)
    detector_grid = (
        detector_targets[0]
        .reshape(renderer.detector.height, renderer.detector.width, 3)
        .detach().cpu().numpy().astype(np.float64)
    )

    detector_origin = detector_grid[0, 0]
    row_step = detector_grid[1, 0] - detector_origin
    column_step = detector_grid[0, 1] - detector_origin
    plane_normal = np.cross(row_step, column_step)

    # Intersect each source-to-vertex ray with the detector plane.
    rays = vertices_world - source
    denominator = rays @ plane_normal
    numerator = float(np.dot(detector_origin - source, plane_normal))
    valid_vertex = np.abs(denominator) > 1e-12
    ray_scale = np.zeros_like(denominator)
    ray_scale[valid_vertex] = numerator / denominator[valid_vertex]
    valid_vertex &= ray_scale > 0.0
    intersections = source + ray_scale[:, None] * rays

    detector_offsets = intersections - detector_origin
    pixel_y = (detector_offsets @ row_step) / np.dot(row_step, row_step)
    pixel_x = (detector_offsets @ column_step) / np.dot(
        column_step, column_step
    )
    projected = np.column_stack((pixel_x, pixel_y))

    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    edge_valid = valid_vertex[edges].all(axis=1)
    edges = edges[edge_valid]
    segments = projected[edges]

    width = renderer.detector.width
    height = renderer.detector.height
    near_image = (
        (segments[..., 0] >= -width)
        & (segments[..., 0] <= 2.0 * width)
        & (segments[..., 1] >= -height)
        & (segments[..., 1] <= 2.0 * height)
    ).any(axis=1)
    return segments[near_image]


def project_stl_axes(stl_path, stl_scale, params, renderer, axis_length_fraction=0.18):
    """Project a small RGB model-axis triad into detector pixel coordinates."""
    from diffdrr.pose import convert

    mesh = trimesh.load(stl_path, force="mesh")
    if stl_scale != 1.0:
        mesh.apply_scale(stl_scale)

    axis_length = float(max(mesh.bounding_box.extents)) * float(axis_length_fraction)
    axis_points_world = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )

    angles_zyx = torch.stack((params[2], params[1], params[0])).unsqueeze(0)
    translation = params[3:].unsqueeze(0)
    pose = convert(
        angles_zyx,
        translation,
        parameterization="euler_angles",
        convention="ZYX",
    )

    with torch.no_grad():
        source, detector_targets = renderer.detector(pose, calibration=None)
    source = source[0, 0].detach().cpu().numpy().astype(np.float64)
    detector_grid = (
        detector_targets[0]
        .reshape(renderer.detector.height, renderer.detector.width, 3)
        .detach().cpu().numpy().astype(np.float64)
    )

    detector_origin = detector_grid[0, 0]
    row_step = detector_grid[1, 0] - detector_origin
    column_step = detector_grid[0, 1] - detector_origin
    plane_normal = np.cross(row_step, column_step)

    rays = axis_points_world - source
    denominator = rays @ plane_normal
    numerator = float(np.dot(detector_origin - source, plane_normal))
    valid = np.abs(denominator) > 1e-12
    ray_scale = np.zeros_like(denominator)
    ray_scale[valid] = numerator / denominator[valid]
    valid &= ray_scale > 0.0
    intersections = source + ray_scale[:, None] * rays

    detector_offsets = intersections - detector_origin
    pixel_y = (detector_offsets @ row_step) / np.dot(row_step, row_step)
    pixel_x = (detector_offsets @ column_step) / np.dot(
        column_step, column_step
    )
    projected = np.column_stack((pixel_x, pixel_y))

    if not bool(valid[0]):
        return []

    axis_specs = (
        ("X", "#FF5252", 1),
        ("Y", "#4CFF68", 2),
        ("Z", "#4DA3FF", 3),
    )
    segments = []
    for label, color, endpoint_index in axis_specs:
        if not bool(valid[endpoint_index]):
            continue
        segments.append(
            {
                "label": label,
                "color": color,
                "segment_xy": np.vstack((projected[0], projected[endpoint_index])),
            }
        )
    return segments


def _add_stl_diagnostic_overlay(ax, target_np, detail_display, wire_segments,
                                axis_segments=None, title=None):
    """Draw raw X-ray attenuation and all STL mesh edges over the ROI."""
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe

    ax.imshow(target_np, cmap="gray")
    visible_detail = np.ma.masked_less_equal(detail_display, 0.02)
    ax.imshow(
        visible_detail,
        cmap="viridis",
        alpha=0.30,
        vmin=0.0,
        vmax=1.0,
    )
    if len(wire_segments):
        lines = LineCollection(
            wire_segments,
            colors="#00E5FF",
            linewidths=0.35,
            alpha=0.32,
        )
        ax.add_collection(lines)
    if axis_segments:
        origin = np.asarray(axis_segments[0]["segment_xy"][0], dtype=np.float64)
        ax.scatter(
            [origin[0]],
            [origin[1]],
            s=26,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )
        for axis in axis_segments:
            segment = np.asarray(axis["segment_xy"], dtype=np.float64)
            color = axis["color"]
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color="white",
                linewidth=3.8,
                alpha=0.85,
                solid_capstyle="round",
                zorder=6,
            )
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linewidth=2.4,
                solid_capstyle="round",
                zorder=7,
            )
            endpoint = segment[1]
            text = ax.text(
                endpoint[0] + 5.0,
                endpoint[1] + 5.0,
                axis["label"],
                color=color,
                fontsize=11,
                weight="bold",
                zorder=8,
            )
            text.set_path_effects(
                [pe.withStroke(linewidth=2.5, foreground="black", alpha=0.75)]
            )
    ax.set_xlim(-0.5, target_np.shape[1] - 0.5)
    ax.set_ylim(target_np.shape[0] - 0.5, -0.5)
    if title is None:
        title = "Final STL Detail Diagnostic\nattenuation + full wireframe"
    ax.set_title(title)
    ax.axis("off")


def save_results(params, optimized_params, optimized_ncc, saved_pose_ncc,
                 final_rx_offset_deg, target_np, drr_np, detail_drr_np,
                 out_dir, renderer, stl_center, stl_path, stl_scale,
                 warm_start_history=None, roi_xywh=None,
                 segmentation_diagnostic=None):
    """Save comparison image, DRR, and pose JSON."""
    os.makedirs(out_dir, exist_ok=True)
    (
        camera_to_centered_stl,
        stl_to_camera,
        centered_stl_to_screen_centered,
        stl_to_screen_centered,
        detector_center_camera,
        model_xyz_deg,
    ) = compute_pose_outputs(params, renderer, stl_center)
    wire_segments = project_stl_wireframe(
        stl_path, stl_scale, params, renderer, stl_center
    )
    axis_segments = project_stl_axes(
        stl_path, stl_scale, params, renderer
    )

    # --- Comparison figure ---
    drr_display = _normalise_for_display(drr_np)
    detail_display = _normalise_for_display(detail_drr_np)
    diff = np.abs(target_np - drr_display)
    projected_mask, projected_box = _projected_silhouette_mask_and_box(
        drr_display
    )
    box_metrics = _box_comparison_metrics(
        projected_box, projected_mask, segmentation_diagnostic
    )
    edge_diagnostic = _compute_edge_diagnostic(
        target_np, projected_mask, segmentation_diagnostic
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.ravel()
    axes[0].imshow(target_np, cmap="gray")
    axes[0].set_title("Radiograph ROI")
    axes[0].axis("off")

    _draw_segmentation_diagnostic(
        axes[1], target_np, segmentation_diagnostic
    )

    _draw_projected_box_diagnostic(
        axes[2],
        target_np,
        drr_display,
        projected_mask,
        projected_box,
        segmentation_diagnostic,
        box_metrics,
    )

    axes[3].imshow(drr_display, cmap="gray")
    axes[3].set_title(f"DRR (saved-pose NCC={saved_pose_ncc:.4f})")
    axes[3].axis("off")

    axes[4].imshow(target_np, cmap="gray")
    axes[4].imshow(drr_display, cmap="magma", alpha=0.45)
    axes[4].set_title("DRR Overlay on ROI")
    axes[4].axis("off")

    _draw_edge_overlay(axes[5], target_np, edge_diagnostic)

    _add_stl_diagnostic_overlay(
        axes[6], target_np, detail_display, wire_segments, axis_segments
    )

    axes[7].imshow(diff, cmap="hot")
    axes[7].set_title("Absolute Difference")
    axes[7].axis("off")

    _draw_metrics_summary(
        axes[8],
        params,
        saved_pose_ncc,
        segmentation_diagnostic,
        box_metrics,
        edge_diagnostic,
        warm_start_history,
    )

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "registration_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved comparison image: {fig_path}")

    # --- High-resolution diagnostic overlay (visualisation only) ---
    diagnostic_fig, diagnostic_ax = plt.subplots(1, 1, figsize=(8, 8))
    _add_stl_diagnostic_overlay(
        diagnostic_ax, target_np, detail_display, wire_segments, axis_segments
    )
    diagnostic_fig.tight_layout()
    diagnostic_path = os.path.join(out_dir, "final_stl_diagnostic.png")
    diagnostic_fig.savefig(diagnostic_path, dpi=200, bbox_inches="tight")
    plt.close(diagnostic_fig)
    print(f"[INFO] Saved final STL diagnostic: {diagnostic_path}")

    # --- Final DRR ---
    drr_img = (drr_display * 255).round().astype(np.uint8)
    drr_path = os.path.join(out_dir, "final_drr.png")
    Image.fromarray(drr_img).save(drr_path)
    print(f"[INFO] Saved DRR: {drr_path}")

    # --- Pose JSON ---
    rx_deg = np.degrees(params[0].item())
    ry_deg = np.degrees(params[1].item())
    rz_deg = np.degrees(params[2].item())
    tx = params[3].item()
    ty = params[4].item()
    tz = params[5].item()
    optimized_rx_deg = np.degrees(optimized_params[0].item())
    optimized_ry_deg = np.degrees(optimized_params[1].item())
    optimized_rz_deg = np.degrees(optimized_params[2].item())

    pose_data = {
        "optimized_diffdrr_camera_parameters": {
            "rotation_xyz_deg": {
                "rx": optimized_rx_deg,
                "ry": optimized_ry_deg,
                "rz": optimized_rz_deg,
            },
            "translation_parameters_mm": {
                "tx": optimized_params[3].item(),
                "ty": optimized_params[4].item(),
                "tz": optimized_params[5].item(),
            },
            "description": "Unmodified NCC-optimized pose",
        },
        "diffdrr_camera_parameters": {
            "rotation_xyz_deg": {"rx": rx_deg, "ry": ry_deg, "rz": rz_deg},
            "translation_parameters_mm": {"tx": tx, "ty": ty, "tz": tz},
            "camera_to_centered_stl_matrix_4x4": camera_to_centered_stl.tolist(),
            "description": (
                "Final saved DiffDRR pose after applying final_rx_offset_deg"
            ),
        },
        "stl_to_camera": {
            "rotation_xyz_deg": {
                "rx": float(model_xyz_deg[0]),
                "ry": float(model_xyz_deg[1]),
                "rz": float(model_xyz_deg[2]),
            },
            "rotation_matrix_3x3": stl_to_camera[:3, :3].tolist(),
            "translation_mm": {
                "tx": float(stl_to_camera[0, 3]),
                "ty": float(stl_to_camera[1, 3]),
                "tz": float(stl_to_camera[2, 3]),
            },
            "matrix_4x4": stl_to_camera.tolist(),
            "description": (
                "Requested rigid transform from original STL coordinates to "
                "DiffDRR camera coordinates (source at origin, detector along +Z)"
            ),
        },
        "stl_to_screen_centered": {
            "rotation_xyz_deg": {
                "rx": float(model_xyz_deg[0]),
                "ry": float(model_xyz_deg[1]),
                "rz": float(model_xyz_deg[2]),
            },
            "rotation_matrix_3x3": stl_to_screen_centered[:3, :3].tolist(),
            "translation_mm": {
                "tx": float(stl_to_screen_centered[0, 3]),
                "ty": float(stl_to_screen_centered[1, 3]),
                "tz": float(stl_to_screen_centered[2, 3]),
            },
            "matrix_4x4": stl_to_screen_centered.tolist(),
            "detector_center_in_camera_mm": [
                float(value) for value in detector_center_camera
            ],
            "description": (
                "Rigid transform from original STL coordinates to a "
                "screen-centred coordinate frame. Origin (0, 0, 0) is the "
                "middle of the detector/image plane; axes match DiffDRR camera "
                "coordinates, with +Z from X-ray source toward detector."
            ),
        },
        "centered_stl_to_screen_centered": {
            "rotation_xyz_deg": {
                "rx": float(model_xyz_deg[0]),
                "ry": float(model_xyz_deg[1]),
                "rz": float(model_xyz_deg[2]),
            },
            "rotation_matrix_3x3": centered_stl_to_screen_centered[
                :3, :3
            ].tolist(),
            "translation_mm": {
                "tx": float(centered_stl_to_screen_centered[0, 3]),
                "ty": float(centered_stl_to_screen_centered[1, 3]),
                "tz": float(centered_stl_to_screen_centered[2, 3]),
            },
            "matrix_4x4": centered_stl_to_screen_centered.tolist(),
            "description": (
                "Teacher-friendly 6-DoF pose. Source origin is the centred "
                "STL/object origin used by DiffDRR, not the STL file's CAD "
                "origin. Destination origin (0, 0, 0) is the middle of the "
                "detector/image plane, so the translation is independent of "
                "where the STL file originally started."
            ),
        },
        "stl_center_mm": [float(value) for value in stl_center],
        "final_rx_offset_deg": float(final_rx_offset_deg),
        "optimized_ncc": float(optimized_ncc),
        "saved_pose_ncc": float(saved_pose_ncc),
        "final_ncc": float(saved_pose_ncc),
    }
    if warm_start_history is not None:
        pose_data["warm_start_history"] = warm_start_history
    if roi_xywh is not None:
        pose_data["roi_xywh"] = [int(value) for value in roi_xywh]
    if segmentation_diagnostic is not None:
        pose_data["segmentation_diagnostic"] = {
            "mode": segmentation_diagnostic["mode"],
            "label": segmentation_diagnostic["label"],
            "implant_percentile": float(
                segmentation_diagnostic["implant_percentile"]
            ),
            "bbox_xywh": [
                int(value) for value in segmentation_diagnostic["bbox_xywh"]
            ],
            "oriented_box": _serialise_box(segmentation_diagnostic["box"]),
            "mask_area_px": int(segmentation_diagnostic["mask_area_px"]),
            "roi_area_px": int(segmentation_diagnostic["roi_area_px"]),
        }
    pose_data["final_projection_diagnostics"] = {
        "projected_silhouette_box": (
            None if projected_box is None else _serialise_box(projected_box)
        ),
        "box_comparison_metrics": box_metrics,
        "edge_overlay_metrics": edge_diagnostic["metrics"],
    }
    json_path = os.path.join(out_dir, "optimal_pose.json")
    with open(json_path, "w") as f:
        json.dump(pose_data, f, indent=2)
    print(f"[INFO] Saved pose JSON: {json_path}")

    return (
        rx_deg,
        ry_deg,
        rz_deg,
        tx,
        ty,
        tz,
        stl_to_camera,
        centered_stl_to_screen_centered,
        stl_to_screen_centered,
        model_xyz_deg,
    )


# ---------------------------------------------------------------------------
# 10.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2D-3D Knee Registration (DiffDRR + PyTorch)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--stl", required=True, help="Path to STL model")
    parser.add_argument("--xray", required=True, help="Path to radiograph (PNG/JPG)")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        default=None,
                        help="ROI on radiograph (x y width height).\n"
                             "If omitted, an interactive selector opens.")
    parser.add_argument("--voxel_size", type=float, default=1.0,
                        help="Voxelisation resolution in mm (default: 1.0)")
    parser.add_argument(
        "--stl_scale", type=float, default=1.0,
        help=("Uniform STL unit conversion before voxelisation "
              "(default: 1.0; use 10 if STL coordinates are cm)"),
    )
    parser.add_argument("--sdd", type=float, default=1020.0,
                        help="Source-to-detector distance in mm (default: 1020)")
    parser.add_argument("--pixel_spacing", type=float, default=0.194,
                        help="Detector pixel pitch in mm (default: 0.194)")
    parser.add_argument(
        "--scale_length_cm", type=float, default=None,
        help=("Known X-ray ruler length in cm; opens a two-point selector "
              "and overrides --pixel_spacing"),
    )
    parser.add_argument(
        "--scale_points", type=float, nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Optional repeatable scale endpoints; requires --scale_length_cm",
    )
    parser.add_argument(
        "--auto_fit_no_ruler",
        action="store_true",
        help=(
            "Without a ruler, infer an effective image-fit pixel spacing at a "
            "nominal valid depth, then fit ty to segmentation/ROI. The inferred "
            "spacing is not a physical calibration."
        ),
    )
    parser.add_argument(
        "--projection_mode",
        choices=("silhouette", "attenuation"),
        default="silhouette",
        help=("Projection used for registration: filled external silhouette "
              "or raw attenuation DRR (default: silhouette)"),
    )
    parser.add_argument(
        "--silhouette_threshold", type=float, default=0.01,
        help="Relative ray-hit threshold for the silhouette (default: 0.01)",
    )
    parser.add_argument(
        "--silhouette_blur_sigma", type=float, default=1.5,
        help="Silhouette edge Gaussian sigma in pixels (default: 1.5)",
    )
    parser.add_argument("--init_rx", type=float, default=0.0,
                        help="Initial rotation X in degrees (default: 0)")
    parser.add_argument("--init_ry", type=float, default=0.0,
                        help="Initial rotation Y in degrees (default: 0)")
    parser.add_argument("--init_rz", type=float, default=0.0,
                        help="Initial rotation Z in degrees (default: 0)")
    parser.add_argument(
        "--final_rx_offset_deg", type=float, default=0.0,
        help=("X rotation added only after NCC optimisation for final "
              "visualisations and output transform (default: 0)"),
    )
    parser.add_argument("--init_tx", type=float, default=0.0,
                        help="Initial translation X in mm (default: 0)")
    parser.add_argument(
        "--init_ty", type=float, default=None,
        help=("Manual initial depth/source-to-object distance in mm. "
              "If omitted, estimate it with a silhouette bounding-box sweep."),
    )
    parser.add_argument("--init_tz", type=float, default=0.0,
                        help="Initial translation Z in mm (default: 0)")
    parser.add_argument(
        "--ty_search_steps", type=int, default=13,
        help="Number of automatic initial-ty candidates (default: 13)",
    )
    parser.add_argument(
        "--ty_search_min", type=float, default=None,
        help="Optional minimum automatic ty in mm (default: 0.65 * SDD)",
    )
    parser.add_argument(
        "--ty_search_max", type=float, default=None,
        help="Optional maximum automatic ty in mm (default: 0.97 * SDD)",
    )
    parser.add_argument(
        "--implant_percentile", type=float, default=80.0,
        help="Brightness percentile for implant segmentation (default: 80)",
    )
    parser.add_argument("--coarse_iters", type=int, default=100,
                        help="Coarse-stage iterations (default: 100)")
    parser.add_argument("--fine_iters", type=int, default=300,
                        help="Fine-stage iterations (default: 300)")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Rotation learning rate in radians (default: 5e-3)")
    parser.add_argument(
        "--translation_lr", type=float, default=0.5,
        help="Translation/depth learning rate in mm (default: 0.5)",
    )
    parser.add_argument("--grid_search", action="store_true",
                        help="Run coarse grid search for initialisation")
    parser.add_argument("--grid_angles", type=int, default=5,
                        help="Grid search angular resolution (default: 5)")
    parser.add_argument("--grid_range", type=float, default=30.0,
                        help="Grid search angular range in degrees (default: 30)")
    parser.add_argument("--device", default="auto",
                        help="Device: 'auto', 'cpu', or 'cuda' (default: auto)")
    parser.add_argument("--out_dir", default="./results",
                        help="Output directory (default: ./results)")
    parser.add_argument(
        "--no_initial_preview", action="store_true",
        help="Skip the initial STL-on-ROI preview window",
    )
    parser.add_argument(
        "--preview_only", action="store_true",
        help="Render the initial alignment and exit without optimisation",
    )
    # --- Edge NCC loss (Solution 1B) ---
    parser.add_argument(
        "--edge_ncc_weight", type=float, default=0.3,
        help="Weight of edge-gradient NCC term in combined loss (default: 0.3)",
    )
    # --- rx polish sweep (Solution 1A) ---
    parser.add_argument(
        "--rx_polish_range", type=float, default=6.0,
        help="rx polish sweep range in degrees (default: 6.0)",
    )
    parser.add_argument(
        "--rx_polish_step", type=float, default=0.25,
        help="rx polish sweep step in degrees (default: 0.25)",
    )
    # --- Chamfer-distance refinement (Solution 2D) ---
    parser.add_argument(
        "--chamfer_refine", action="store_true",
        help="Enable Chamfer-distance refinement after rx polish",
    )
    parser.add_argument(
        "--chamfer_ty_range", type=float, default=30.0,
        help="Chamfer ty search range in mm (default: 30.0)",
    )
    parser.add_argument(
        "--chamfer_ty_step", type=float, default=2.0,
        help="Chamfer ty search step in mm (default: 2.0)",
    )
    parser.add_argument(
        "--chamfer_rx_range", type=float, default=4.0,
        help="Chamfer rx search range in degrees (default: 4.0)",
    )
    parser.add_argument(
        "--chamfer_rx_step", type=float, default=0.25,
        help="Chamfer rx search step in degrees (default: 0.25)",
    )
    parser.add_argument(
        "--chamfer_segment_mode",
        choices=("implant", "bone"),
        default="implant",
        help=(
            "Segmentation used only by Chamfer refinement: bright implant "
            "percentile mask or Otsu bone mask (default: implant)"
        ),
    )
    parser.add_argument(
        "--chamfer_combined_objective",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rank Chamfer candidates with normalized bidirectional Chamfer + "
            "mask-IoU loss + outside-target STL penalty. Use "
            "--no-chamfer_combined_objective for pure Chamfer (default: off)"
        ),
    )
    parser.add_argument(
        "--chamfer_iou_weight", type=float, default=1.0,
        help="Weight of (1 - mask IoU) in the combined contour objective",
    )
    parser.add_argument(
        "--chamfer_outside_weight", type=float, default=1.0,
        help=(
            "Weight of the projected-STL outside-target fraction in the "
            "combined contour objective"
        ),
    )
    # --- Segmentation mode (Solution 2B) ---
    parser.add_argument(
        "--segment_mode", choices=("bone", "implant"), default="implant",
        help=(
            "Target segmentation for automatic depth calibration: Otsu bone "
            "or bright implant (default: implant). Use bone only when the ROI "
            "is a true bone silhouette rather than a metal implant."
        ),
    )
    parser.add_argument(
        "--warm_start_passes", type=int, default=1,
        help=("Extra warm-start refinement passes. The current best pose is "
              "fed back through the optimisation pipeline and kept only if "
              "NCC improves (default: 1)"),
    )
    parser.add_argument(
        "--warm_start_min_delta", type=float, default=1e-4,
        help=("Minimum NCC improvement required to accept a warm-start pass "
              "(default: 1e-4)"),
    )
    parser.add_argument(
        "--warm_start_free_ty", action="store_true",
        help=("Allow warm-start passes to optimise ty/depth. By default ty is "
              "locked during warm starts to prevent repeated scale growth."),
    )
    parser.add_argument(
        "--ty_guard_mm", type=float, default=8.0,
        help=("Maximum allowed deviation from the calibrated initial ty during "
              "optimisation/refinement. Use <=0 to disable (default: 8.0)"),
    )
    args = parser.parse_args()

    if not 0.0 < args.silhouette_threshold < 1.0:
        parser.error("--silhouette_threshold must be between 0 and 1")
    if args.silhouette_blur_sigma < 0.0:
        parser.error("--silhouette_blur_sigma must be non-negative")
    if args.preview_only and args.no_initial_preview:
        parser.error("--preview_only cannot be combined with --no_initial_preview")
    if args.stl_scale <= 0.0:
        parser.error("--stl_scale must be positive")
    if args.scale_length_cm is not None and args.scale_length_cm <= 0.0:
        parser.error("--scale_length_cm must be positive")
    if args.scale_points is not None and args.scale_length_cm is None:
        parser.error("--scale_points requires --scale_length_cm")
    if args.scale_length_cm is None and args.pixel_spacing <= 0.0:
        parser.error(
            "--pixel_spacing must be positive when no scale ruler is provided"
        )
    if args.translation_lr <= 0.0:
        parser.error("--translation_lr must be positive")
    if args.ty_search_steps < 3:
        parser.error("--ty_search_steps must be at least 3")
    if not 50.0 <= args.implant_percentile < 100.0:
        parser.error("--implant_percentile must be in [50, 100)")
    if args.ty_search_min is not None and args.ty_search_min <= 0.0:
        parser.error("--ty_search_min must be positive")
    if args.ty_search_max is not None and args.ty_search_max <= 0.0:
        parser.error("--ty_search_max must be positive")
    if (
        args.ty_search_min is not None
        and args.ty_search_max is not None
        and args.ty_search_min >= args.ty_search_max
    ):
        parser.error("--ty_search_min must be smaller than --ty_search_max")
    if args.warm_start_passes < 0:
        parser.error("--warm_start_passes must be non-negative")
    if args.warm_start_min_delta < 0.0:
        parser.error("--warm_start_min_delta must be non-negative")
    if args.chamfer_ty_range < 0.0 or args.chamfer_rx_range < 0.0:
        parser.error("Chamfer search ranges must be non-negative")
    if args.chamfer_ty_step <= 0.0 or args.chamfer_rx_step <= 0.0:
        parser.error("Chamfer search steps must be positive")
    if args.chamfer_iou_weight < 0.0:
        parser.error("--chamfer_iou_weight must be non-negative")
    if args.chamfer_outside_weight < 0.0:
        parser.error("--chamfer_outside_weight must be non-negative")
    if args.ty_guard_mm < 0.0:
        print("[WARN] --ty_guard_mm < 0 disables the calibrated depth guard")

    # --- Device ---
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {args.out_dir}")

    # --- Load STL and voxelize ---
    volume, affine, stl_center = load_and_voxelize(
        args.stl, args.voxel_size, args.stl_scale
    )

    # --- Load radiograph and select ROI ---
    print(f"[INFO] Loading radiograph: {args.xray}")
    xray_np = load_image_as_array(args.xray)
    print(f"[INFO] Radiograph size: {xray_np.shape}")

    effective_pixel_spacing = args.pixel_spacing
    if args.scale_length_cm is not None:
        effective_pixel_spacing = calibrate_pixel_spacing(
            xray_np,
            args.scale_length_cm,
            args.out_dir,
            points=args.scale_points,
        )
    else:
        # Reusing an output directory must not make a no-ruler run appear to
        # have performed physical scale calibration. These two files are
        # generated artifacts, so remove stale copies from an earlier run.
        for filename in ("scale_calibration.png", "scale_calibration.json"):
            stale_path = os.path.join(args.out_dir, filename)
            if os.path.isfile(stale_path):
                os.remove(stale_path)
                print(
                    "[INFO] Removed stale ruler-calibration artifact from "
                    f"a previous run: {stale_path}"
                )
        print(
            "[INFO] No X-ray ruler calibration: using configured detector "
            f"pixel spacing {effective_pixel_spacing:.6f} mm/pixel"
        )

    if args.roi is not None:
        roi_x, roi_y, roi_w, roi_h = args.roi
        print(f"[INFO] Using CLI ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
    else:
        roi_x, roi_y, roi_w, roi_h = interactive_roi_select(xray_np)
        print(f"[INFO] Using interactive ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

    image_height, image_width = xray_np.shape
    if (
        roi_w <= 0 or roi_h <= 0
        or roi_x < 0 or roi_y < 0
        or roi_x + roi_w > image_width
        or roi_y + roi_h > image_height
    ):
        raise ValueError(
            f"ROI {(roi_x, roi_y, roi_w, roi_h)} is outside image bounds "
            f"(width={image_width}, height={image_height})"
        )

    # Crop and preprocess
    roi_img = xray_np[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_processed = preprocess_roi(roi_img)
    target = torch.from_numpy(roi_processed).float().to(device)
    print(f"[INFO] Radiograph ROI tensor shape: {target.shape}")

    segmentation_diagnostic = None
    try:
        segmentation_diagnostic = build_segmentation_diagnostic(
            roi_img, args.segment_mode, args.implant_percentile
        )
    except Exception as error:
        print(
            f"[WARN] Could not build segment_mode diagnostic panel ({error}); "
            "registration_comparison.png will mark it as unavailable."
        )

    chamfer_segmentation = None
    if args.chamfer_refine:
        try:
            if (
                segmentation_diagnostic is not None
                and segmentation_diagnostic["mode"]
                == args.chamfer_segment_mode
            ):
                chamfer_segmentation = segmentation_diagnostic
            else:
                chamfer_segmentation = build_segmentation_diagnostic(
                    roi_img,
                    args.chamfer_segment_mode,
                    args.implant_percentile,
                )
            print(
                "[INFO] Chamfer segmentation prepared: "
                f"{chamfer_segmentation['label']}"
            )
        except Exception as error:
            print(
                "[WARN] Could not prepare the requested Chamfer "
                f"segmentation ({error}); Chamfer will try a processed-ROI "
                "fallback."
            )

    # --- Build DiffDRR renderer ---
    detector_height, detector_width = roi_img.shape
    renderer = build_renderer(
        volume, affine, args.sdd, effective_pixel_spacing,
        detector_height, detector_width, device,
        projection_mode=args.projection_mode,
        silhouette_threshold=args.silhouette_threshold,
        silhouette_blur_sigma=args.silhouette_blur_sigma,
    )

    # --- Initialisation ---
    initial_ty = args.init_ty
    base_ty = initial_ty if initial_ty is not None else 0.85 * args.sdd
    base_params = torch.tensor(
        [
            np.radians(args.init_rx),
            np.radians(args.init_ry),
            np.radians(args.init_rz),
            args.init_tx,
            base_ty,
            args.init_tz,
        ],
        dtype=torch.float32,
        device=device,
    )

    if (
        initial_ty is None
        and args.scale_length_cm is None
        and args.auto_fit_no_ruler
    ):
        try:
            effective_pixel_spacing, no_ruler_fit = estimate_no_ruler_pixel_spacing(
                renderer,
                roi_img,
                base_params,
                device,
                effective_pixel_spacing,
                args.stl,
                args.stl_scale,
                stl_center,
                implant_percentile=args.implant_percentile,
                segment_mode=args.segment_mode,
            )
            renderer = build_renderer(
                volume, affine, args.sdd, effective_pixel_spacing,
                detector_height, detector_width, device,
                projection_mode=args.projection_mode,
                silhouette_threshold=args.silhouette_threshold,
                silhouette_blur_sigma=args.silhouette_blur_sigma,
            )
            os.makedirs(args.out_dir, exist_ok=True)
            no_ruler_fit_path = os.path.join(
                args.out_dir, "no_ruler_image_fit.json"
            )
            with open(no_ruler_fit_path, "w") as f:
                json.dump(no_ruler_fit, f, indent=2)
            print(f"[INFO] Saved no-ruler image-fit data: {no_ruler_fit_path}")
        except Exception as error:
            print(
                "[WARN] Could not infer no-ruler image-fit spacing "
                f"({error}); continuing with configured pixel spacing"
            )

    if initial_ty is None:
        if args.scale_length_cm is None:
            print(
                "[INFO] No ruler: fitting initial depth directly to the target "
                "segmentation across the full valid depth range"
            )
            try:
                initial_ty = estimate_initial_ty(
                    renderer, roi_img, base_params, device, args.sdd,
                    args.out_dir,
                    steps=args.ty_search_steps,
                    ty_min=args.ty_search_min,
                    ty_max=args.ty_search_max,
                    implant_percentile=args.implant_percentile,
                    segment_mode=args.segment_mode,
                    stl_path=args.stl,
                    stl_scale=args.stl_scale,
                    stl_center=stl_center,
                )
            except Exception as error:
                initial_ty = 0.85 * args.sdd
                print(
                    f"[WARN] No-ruler depth fitting failed ({error}); "
                    f"fallback ty={initial_ty:.3f} mm"
                )
        else:
            # With a ruler, seed depth from physical magnification and refine it.
            try:
                if args.segment_mode == "bone":
                    bone_bbox, bone_mask = segment_bone_bbox(roi_img)
                else:
                    bone_bbox, bone_mask = segment_implant_bbox(
                        roi_img, percentile=args.implant_percentile
                    )
                target_box = _mask_box(bone_mask)
                if target_box is None:
                    target_box = _box_from_bbox(bone_bbox)
                initial_ty = closed_form_ty(
                    args.stl, args.stl_scale,
                    target_box, effective_pixel_spacing, args.sdd
                )
                try:
                    initial_ty = estimate_initial_ty(
                        renderer, roi_img, base_params, device, args.sdd,
                        args.out_dir,
                        steps=args.ty_search_steps,
                        ty_min=initial_ty * 0.85,
                        ty_max=initial_ty * 1.15,
                        implant_percentile=args.implant_percentile,
                        segment_mode=args.segment_mode,
                        stl_path=args.stl,
                        stl_scale=args.stl_scale,
                        stl_center=stl_center,
                    )
                except Exception:
                    pass  # closed-form estimate is good enough
            except Exception as error:
                print(f"[WARN] Closed-form ty failed ({error}); using full sweep")
                try:
                    initial_ty = estimate_initial_ty(
                        renderer, roi_img, base_params, device, args.sdd,
                        args.out_dir,
                        steps=args.ty_search_steps,
                        ty_min=args.ty_search_min,
                        ty_max=args.ty_search_max,
                        implant_percentile=args.implant_percentile,
                        segment_mode=args.segment_mode,
                        stl_path=args.stl,
                        stl_scale=args.stl_scale,
                        stl_center=stl_center,
                    )
                except Exception as error2:
                    initial_ty = 0.85 * args.sdd
                    print(
                        f"[WARN] ty estimation failed ({error2}); "
                        f"fallback ty={initial_ty:.3f} mm"
                    )
    else:
        print(
            f"[INFO] Manual --init_ty override: {initial_ty:.3f} mm "
            "(automatic depth estimation skipped)"
        )

    init_params = torch.tensor(
        [
            np.radians(args.init_rx),
            np.radians(args.init_ry),
            np.radians(args.init_rz),
            args.init_tx,
            initial_ty,
            args.init_tz,
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    print(f"[INFO] Initial pose: rx={args.init_rx} deg ry={args.init_ry} deg rz={args.init_rz} deg "
          f"tx={args.init_tx} ty={initial_ty} tz={args.init_tz}")

    if args.grid_search:
        init_params = coarse_grid_search(
            renderer, target, init_params, device,
            n_angles=args.grid_angles,
            angle_range_deg=args.grid_range,
        )

    # --- Show the initial STL pose before optimisation ---
    if not args.no_initial_preview:
        should_continue = preview_initial_alignment(
            renderer,
            target,
            init_params,
            device,
            args.out_dir,
            args.stl,
            args.stl_scale,
            stl_center,
            show_window=True,
            ask_to_continue=not args.preview_only,
        )
        if args.preview_only:
            print("[INFO] Preview-only mode complete; optimisation was not run.")
            return
        if not should_continue:
            print("[INFO] Optimisation cancelled after the initial preview.")
            return

    # --- Optimise / warm start ---
    best_params, optimized_ncc, chamfer_dist, warm_start_history = (
        run_warm_start_pipeline(
            renderer,
            target,
            init_params,
            device,
            args,
            effective_pixel_spacing,
            args.stl,
            args.stl_scale,
            stl_center,
            chamfer_segmentation=chamfer_segmentation,
        )
    )

    # Apply an optional anatomical/visual correction only after optimisation.
    # best_params remains the exact pose selected by NCC.
    final_params = best_params.detach().clone()
    final_params[0] += np.radians(args.final_rx_offset_deg)
    if args.final_rx_offset_deg != 0.0:
        print(
            f"[INFO] Applying final-only rx offset: "
            f"{args.final_rx_offset_deg:+.3f} deg"
        )

    # --- Render final DRR ---
    with torch.no_grad():
        final_drr = render_drr(renderer, final_params, device)
        # Diagnostic-only attenuation rendering. This is never used by NCC,
        # backpropagation, parameter selection, or the optimizer.
        final_detail_drr = render_drr(
            renderer, final_params, device, projection_mode="attenuation"
        )
        if final_drr.shape != target.shape:
            final_drr = torch.nn.functional.interpolate(
                final_drr.unsqueeze(0).unsqueeze(0),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        saved_pose_ncc = 1.0 - ncc_loss(final_drr, target).item()
        if final_detail_drr.shape != target.shape:
            final_detail_drr = torch.nn.functional.interpolate(
                final_detail_drr.unsqueeze(0).unsqueeze(0),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

    target_np = target.detach().cpu().numpy()
    drr_np = final_drr.detach().cpu().numpy()
    detail_drr_np = final_detail_drr.detach().cpu().numpy()

    # --- Save results ---
    (
        rx,
        ry,
        rz,
        tx,
        ty,
        tz,
        stl_to_camera,
        centered_stl_to_screen_centered,
        stl_to_screen_centered,
        model_xyz_deg,
    ) = save_results(
        final_params, best_params, optimized_ncc, saved_pose_ncc,
        args.final_rx_offset_deg, target_np, drr_np, detail_drr_np,
        args.out_dir, renderer, stl_center, args.stl, args.stl_scale,
        warm_start_history=warm_start_history,
        roi_xywh=(roi_x, roi_y, roi_w, roi_h),
        segmentation_diagnostic=segmentation_diagnostic,
    )

    # --- Print final transformation ---
    print("\n" + "=" * 60)
    print("  FINAL SAVED RESULT (RAW DIFFDRR CAMERA PARAMETERS)")
    print("=" * 60)
    print(f"  Rotation  rx = {rx:+.3f} deg")
    print(f"            ry = {ry:+.3f} deg")
    print(f"            rz = {rz:+.3f} deg")
    print(f"  Translation tx = {tx:+.3f} mm")
    print(f"            ty = {ty:+.3f} mm")
    print(f"            tz = {tz:+.3f} mm")
    print(f"  Final rx offset = {args.final_rx_offset_deg:+.3f} deg")
    print(f"  Optimized NCC = {optimized_ncc:.6f}")
    print(f"  Saved-pose NCC = {saved_pose_ncc:.6f}")
    if chamfer_dist > 0:
        print(f"  Chamfer distance = {chamfer_dist:.2f} px")
    accepted_warm_starts = sum(
        1 for item in warm_start_history[1:] if item["accepted"]
    )
    attempted_warm_starts = max(0, len(warm_start_history) - 1)
    if attempted_warm_starts > 0:
        print(
            f"  Warm starts accepted = {accepted_warm_starts}/{attempted_warm_starts}"
        )
    print("=" * 60)
    print("  REQUESTED ORIGINAL-STL -> CAMERA TRANSFORM")
    print("=" * 60)
    print(f"  Rotation  rx = {model_xyz_deg[0]:+.3f} deg")
    print(f"            ry = {model_xyz_deg[1]:+.3f} deg")
    print(f"            rz = {model_xyz_deg[2]:+.3f} deg")
    print(f"  Translation tx = {stl_to_camera[0, 3]:+.3f} mm")
    print(f"              ty = {stl_to_camera[1, 3]:+.3f} mm")
    print(f"              tz = {stl_to_camera[2, 3]:+.3f} mm")
    print("=" * 60)
    print("  CENTRED-STL/OBJECT -> SCREEN-CENTRED 6-DOF POSE")
    print("  Origin (0,0,0) = middle of detector/image plane")
    print("=" * 60)
    print(f"  Rotation  rx = {model_xyz_deg[0]:+.3f} deg")
    print(f"            ry = {model_xyz_deg[1]:+.3f} deg")
    print(f"            rz = {model_xyz_deg[2]:+.3f} deg")
    print(f"  Translation tx = {centered_stl_to_screen_centered[0, 3]:+.3f} mm")
    print(f"              ty = {centered_stl_to_screen_centered[1, 3]:+.3f} mm")
    print(f"              tz = {centered_stl_to_screen_centered[2, 3]:+.3f} mm")
    print("  Full rotation and homogeneous matrices are in optimal_pose.json")
    print("=" * 60)
    print(f"\n  Results saved to: {args.out_dir}/")
    if args.scale_length_cm is not None:
        print(f"    - scale_calibration.png")
        print(f"    - scale_calibration.json")
    if args.init_ty is None and os.path.exists(
        os.path.join(args.out_dir, "depth_calibration.json")
    ):
        print(f"    - depth_calibration.png")
        print(f"    - depth_calibration.json")
    if args.chamfer_refine:
        print(f"    - chamfer_refinement.png")
    if not args.no_initial_preview:
        print(f"    - initial_alignment.png")
    print(f"    - registration_comparison.png")
    print(f"    - final_stl_diagnostic.png")
    print(f"    - final_drr.png")
    print(f"    - optimal_pose.json")


if __name__ == "__main__":
    main()
