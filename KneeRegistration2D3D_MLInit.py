#!/usr/bin/env python3
"""
The recommended workflow is deliberately conservative:

1. Manually place each STL on the X-ray ROI with:

       python KneeRegistration2D3D_MLInit.py place --xray Xray/ASIP1.jpg

2. Save those placements as labels.
3. Train a small pose initialiser from those labels.
4. Use the predicted pose only as the initial pose for the geometric
   NCC/Chamfer optimiser in KneeRegistration2D3D_V2.py.

The trained model is an initializer, not the final judge. The final pose should
still be refined and checked by the existing physics/geometric pipeline.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.widgets import Button, Slider, TextBox
from PIL import Image


POSE_KEYS = ("rx", "ry", "rz", "tx", "ty", "tz")
DEFAULT_STLS = (
    "3D_files/Phase3FemurM.stl",
    "3D_files/Phase3TibiaRightB.stl",
)
ROTATION_SLIDER_MIN = -180.0
ROTATION_SLIDER_MAX = 180.0
DEFAULT_FIRST_STL_ROTATION = (0.0, 0.0, 0.0)
DEFAULT_SECOND_STL_ROTATION = (-90.0, 0.0, 0.0)


def disable_conflicting_matplotlib_keymaps() -> None:
    """Keep manual-placement keys from triggering Matplotlib shortcuts."""
    for keymap_name in (
        "keymap.quit",
        "keymap.quit_all",
        "keymap.save",
        "keymap.back",
        "keymap.forward",
    ):
        if keymap_name in matplotlib.rcParams:
            matplotlib.rcParams[keymap_name] = []


def round_pose_value(value: float) -> float:
    """Use 0.1 resolution everywhere in the manual placement UI."""
    return round(float(value), 1)


def slider_bounds(value: float, default_min: float, default_max: float):
    """Return valid slider bounds that always contain the initial value."""
    value = float(value)
    default_min = float(default_min)
    default_max = float(default_max)
    if not np.isfinite([value, default_min, default_max]).all():
        raise ValueError("Slider values and bounds must be finite")
    if default_min >= default_max:
        raise ValueError(
            f"Slider minimum must be below maximum: {default_min} >= {default_max}"
        )
    return min(default_min, value), max(default_max, value)


def initial_rotation_for_stl(
    args: argparse.Namespace,
    stl_index: int,
) -> tuple[float, float, float]:
    """Resolve common and optional second-STL initial rotations."""
    raw_common = (args.init_rx, args.init_ry, args.init_rz)
    common = tuple(
        float(default if value is None else value)
        for value, default in zip(raw_common, DEFAULT_FIRST_STL_ROTATION)
    )
    if stl_index != 1 or args.disable_second_stl_init_override:
        return common

    raw_second = (
        args.second_init_rx,
        args.second_init_ry,
        args.second_init_rz,
    )
    if any(value is not None for value in raw_second):
        return tuple(
            float(common_value if value is None else value)
            for value, common_value in zip(raw_second, common)
        )
    if any(value is not None for value in raw_common):
        return common
    return DEFAULT_SECOND_STL_ROTATION


def load_kreg_module():
    """Import KneeRegistration2D3D_V2.py from the same folder."""
    script_path = Path(__file__).resolve().with_name("KneeRegistration2D3D_V2.py")
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find {script_path}. Put this file next to "
            "KneeRegistration2D3D_V2.py."
        )
    spec = importlib.util.spec_from_file_location("kreg_v2", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def path_for_record(path: str | Path) -> str:
    """Store relative paths when possible, absolute paths otherwise."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def resolve_record_path(record: dict[str, Any], key: str) -> Path:
    """Resolve a path saved in a label record."""
    raw = Path(record[key])
    if raw.is_absolute():
        return raw
    root = Path(record.get("project_root", "."))
    return (root / raw).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def tensor_params_from_pose(
    pose: dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            np.radians(float(pose["rx"])),
            np.radians(float(pose["ry"])),
            np.radians(float(pose["rz"])),
            float(pose["tx"]),
            float(pose["ty"]),
            float(pose["tz"]),
        ],
        dtype=torch.float32,
        device=device,
    )


def pose_from_tensor(params: torch.Tensor) -> dict[str, float]:
    params_cpu = params.detach().cpu()
    return {
        "rx": float(np.degrees(params_cpu[0].item())),
        "ry": float(np.degrees(params_cpu[1].item())),
        "rz": float(np.degrees(params_cpu[2].item())),
        "tx": float(params_cpu[3].item()),
        "ty": float(params_cpu[4].item()),
        "tz": float(params_cpu[5].item()),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def append_csv(path: Path, record: dict[str, Any]) -> None:
    ensure_parent(path)
    pose = record["pose"]
    row = {
        "label_id": record["label_id"],
        "timestamp": record["timestamp"],
        "xray": record["xray"],
        "stl": record["stl"],
        "stl_name": record["stl_name"],
        "roi_x": record["roi_xywh"][0],
        "roi_y": record["roi_xywh"][1],
        "roi_w": record["roi_xywh"][2],
        "roi_h": record["roi_xywh"][3],
        "pixel_spacing_mm": record["pixel_spacing_mm"],
        "sdd": record["sdd"],
        "stl_scale": record["stl_scale"],
        "voxel_size": record["voxel_size"],
        "segment_mode": record["segment_mode"],
        "implant_percentile": record["implant_percentile"],
        "rx": pose["rx"],
        "ry": pose["ry"],
        "rz": pose["rz"],
        "tx": pose["tx"],
        "ty": pose["ty"],
        "tz": pose["tz"],
        "ncc": record.get("ncc"),
        "diagnostic_dir": record.get("diagnostic_dir"),
    }
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_labels(labels_jsonl: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with labels_jsonl.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on {labels_jsonl}:{line_number}: {error}"
                ) from error
    if not records:
        raise ValueError(f"No labels found in {labels_jsonl}")
    return records


def load_roi_array(record: dict[str, Any], image_size: int) -> np.ndarray:
    xray_path = resolve_record_path(record, "xray")
    image = np.asarray(Image.open(xray_path).convert("L"), dtype=np.float32)
    x, y, w, h = [int(v) for v in record["roi_xywh"]]
    roi = image[y : y + h, x : x + w]
    roi = np.clip(roi, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)
    roi = cv2.resize(roi, (image_size, image_size), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32) / 255.0
    return roi[None, :, :]


class PoseLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: int,
        stl_to_id: dict[str, int] | None = None,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ):
        self.records = records
        self.image_size = int(image_size)
        if stl_to_id is None:
            stl_keys = sorted({self.stl_key(record) for record in records})
            stl_to_id = {key: index for index, key in enumerate(stl_keys)}
        self.stl_to_id = stl_to_id
        targets = np.asarray(
            [[float(record["pose"][key]) for key in POSE_KEYS] for record in records],
            dtype=np.float32,
        )
        self.target_mean = (
            targets.mean(axis=0) if target_mean is None else target_mean.astype(np.float32)
        )
        self.target_std = (
            targets.std(axis=0) if target_std is None else target_std.astype(np.float32)
        )
        self.target_std = np.maximum(self.target_std, 1e-3)

    @staticmethod
    def stl_key(record: dict[str, Any]) -> str:
        return str(record.get("stl_name") or Path(record["stl"]).stem)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_roi_array(record, self.image_size)
        stl_key = self.stl_key(record)
        stl_id = self.stl_to_id.get(stl_key, 0)
        target = np.asarray(
            [float(record["pose"][key]) for key in POSE_KEYS],
            dtype=np.float32,
        )
        target = (target - self.target_mean) / self.target_std
        return (
            torch.from_numpy(image).float(),
            torch.tensor(stl_id, dtype=torch.long),
            torch.from_numpy(target).float(),
        )


class TinyPoseCNN(nn.Module):
    """Small CPU-friendly pose regressor for an initial 6-DoF guess."""

    def __init__(self, n_stls: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.stl_embedding = nn.Embedding(max(1, n_stls), 8)
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 6),
        )

    def forward(self, image: torch.Tensor, stl_id: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        e = self.stl_embedding(stl_id)
        return self.head(torch.cat([x, e], dim=1))


class ManualPlacementSession:
    def __init__(
        self,
        *,
        kreg,
        xray_path: Path,
        roi_img: np.ndarray,
        roi_xywh: tuple[int, int, int, int],
        stl_path: Path,
        stl_name: str,
        renderer,
        stl_center: np.ndarray,
        target: torch.Tensor,
        init_params: torch.Tensor,
        device: torch.device,
        args: argparse.Namespace,
        pixel_spacing: float,
        diagnostic_dir: Path,
    ):
        self.kreg = kreg
        self.xray_path = xray_path
        self.roi_img = roi_img
        self.roi_xywh = tuple(int(v) for v in roi_xywh)
        self.stl_path = stl_path
        self.stl_name = stl_name
        self.renderer = renderer
        self.stl_center = stl_center
        self.target = target
        self.init_params = init_params.detach().clone()
        self.device = device
        self.args = args
        self.pixel_spacing = float(pixel_spacing)
        self.diagnostic_dir = diagnostic_dir
        self.saved_record: dict[str, Any] | None = None
        self.last_ncc: float | None = None

        self.target_np = target.detach().cpu().numpy()
        self.fig = None
        self.ax = None
        self.sliders: dict[str, Slider] = {}
        self.text_boxes: dict[str, TextBox] = {}
        self._syncing_widgets = False

    def _slider_value(self, key: str) -> float:
        return float(self.sliders[key].val)

    def expand_slider_range(
        self,
        key: str,
        value: float,
        *,
        expand_at_boundary: bool = False,
    ) -> None:
        """Grow a slider range so manual placement never hits a hard stop."""
        slider = self.sliders[key]
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")

        old_min = float(slider.valmin)
        old_max = float(slider.valmax)
        span = max(old_max - old_min, 1.0)
        headroom = 0.5 * span
        expand_lower = value < old_min or (
            expand_at_boundary and value <= old_min
        )
        expand_upper = value > old_max or (
            expand_at_boundary and value >= old_max
        )
        if not expand_lower and not expand_upper:
            return

        new_min = min(old_min, value - headroom) if expand_lower else old_min
        new_max = max(old_max, value + headroom) if expand_upper else old_max
        slider.valmin = new_min
        slider.valmax = new_max
        if slider.orientation == "vertical":
            slider.ax.set_ylim(new_min, new_max)
        else:
            slider.ax.set_xlim(new_min, new_max)

    def set_slider_value(self, key: str, value: float) -> None:
        slider = self.sliders[key]
        value = round_pose_value(float(value))
        self.expand_slider_range(key, value)
        slider.set_val(value)

    def on_slider_changed(self, key: str, value: float) -> None:
        self.expand_slider_range(key, value, expand_at_boundary=True)
        self.update()

    def sync_text_boxes(self) -> None:
        if self._syncing_widgets:
            return
        self._syncing_widgets = True
        try:
            for key, text_box in self.text_boxes.items():
                value_text = f"{self._slider_value(key):.1f}"
                if getattr(text_box, "text", None) != value_text:
                    text_box.set_val(value_text)
        finally:
            self._syncing_widgets = False

    def on_text_submit(self, key: str, text: str) -> None:
        if self._syncing_widgets:
            return
        try:
            value = float(text)
        except ValueError:
            self.sync_text_boxes()
            return
        self.set_slider_value(key, value)
        self.sync_text_boxes()

    def current_pose(self) -> dict[str, float]:
        return {
            "rx": self._slider_value("rx"),
            "ry": self._slider_value("ry"),
            "rz": self._slider_value("rz"),
            "tx": self._slider_value("tx"),
            "ty": self._slider_value("ty"),
            "tz": self._slider_value("tz"),
        }

    def current_params(self) -> torch.Tensor:
        return tensor_params_from_pose(self.current_pose(), self.device)

    def render_current(self):
        params = self.current_params()
        with torch.no_grad():
            drr = self.kreg.render_drr(self.renderer, params, self.device)
            detail = self.kreg.render_drr(
                self.renderer, params, self.device, projection_mode="attenuation"
            )
            if drr.shape != self.target.shape:
                drr = torch.nn.functional.interpolate(
                    drr.unsqueeze(0).unsqueeze(0),
                    size=self.target.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            if detail.shape != self.target.shape:
                detail = torch.nn.functional.interpolate(
                    detail.unsqueeze(0).unsqueeze(0),
                    size=self.target.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            self.last_ncc = 1.0 - self.kreg.ncc_loss(drr, self.target).item()
        return params, drr, detail

    def update(self, _=None) -> None:
        if self.ax is None:
            return
        params, drr, detail = self.render_current()
        drr_display = self.kreg._normalise_for_display(drr.detach().cpu().numpy())
        detail_display = self.kreg._normalise_for_display(
            detail.detach().cpu().numpy()
        )
        wire_segments = self.kreg.project_stl_wireframe(
            str(self.stl_path),
            self.args.stl_scale,
            params,
            self.renderer,
            self.stl_center,
        )
        axis_segments = self.kreg.project_stl_axes(
            str(self.stl_path),
            self.args.stl_scale,
            params,
            self.renderer,
        )
        self.ax.clear()
        self.kreg._add_stl_diagnostic_overlay(
            self.ax,
            self.target_np,
            detail_display,
            wire_segments,
            axis_segments=axis_segments,
            title=(
                f"Manual placement: {self.stl_name}\n"
                "Place the STL outline on the matching X-ray component"
            ),
        )
        visible_silhouette = np.ma.masked_less_equal(drr_display, 0.05)
        self.ax.imshow(
            visible_silhouette,
            cmap="magma",
            alpha=0.30,
            vmin=0.0,
            vmax=1.0,
        )
        pose = self.current_pose()
        self.ax.text(
            0.02,
            0.98,
            (
                f"NCC={self.last_ncc:.4f}\n"
                f"rx={pose['rx']:+.1f} ry={pose['ry']:+.1f} rz={pose['rz']:+.1f} deg\n"
                f"tx={pose['tx']:+.1f} ty={pose['ty']:+.1f} tz={pose['tz']:+.1f} mm"
            ),
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
        )
        self.sync_text_boxes()
        self.fig.canvas.draw_idle()

    def save(self, _=None) -> None:
        params, drr, detail = self.render_current()
        pose = self.current_pose()
        label_id = (
            f"{Path(self.xray_path).stem}__{self.stl_name}__"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        diag_dir = self.diagnostic_dir / label_id
        segmentation = None
        try:
            segmentation = self.kreg.build_segmentation_diagnostic(
                self.roi_img,
                self.args.segment_mode,
                self.args.implant_percentile,
            )
        except Exception as error:
            print(f"[WARN] Could not compute segmentation diagnostic: {error}")

        target_np = self.target.detach().cpu().numpy()
        drr_np = drr.detach().cpu().numpy()
        detail_np = detail.detach().cpu().numpy()
        saved_pose_ncc = float(self.last_ncc if self.last_ncc is not None else 0.0)

        self.kreg.save_results(
            params,
            params,
            saved_pose_ncc,
            saved_pose_ncc,
            0.0,
            target_np,
            drr_np,
            detail_np,
            str(diag_dir),
            self.renderer,
            self.stl_center,
            str(self.stl_path),
            self.args.stl_scale,
            warm_start_history=[
                {
                    "pass_index": 0,
                    "kind": "manual",
                    "ncc": saved_pose_ncc,
                    "ty_mm": float(pose["ty"]),
                    "accepted": True,
                    "improvement": None,
                }
            ],
            roi_xywh=self.roi_xywh,
            segmentation_diagnostic=segmentation,
        )

        record = {
            "label_id": label_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(Path.cwd().resolve()),
            "xray": path_for_record(self.xray_path),
            "stl": path_for_record(self.stl_path),
            "stl_name": self.stl_name,
            "roi_xywh": list(self.roi_xywh),
            "pixel_spacing_mm": float(self.pixel_spacing),
            "sdd": float(self.args.sdd),
            "stl_scale": float(self.args.stl_scale),
            "voxel_size": float(self.args.voxel_size),
            "segment_mode": self.args.segment_mode,
            "implant_percentile": float(self.args.implant_percentile),
            "pose": pose,
            "ncc": saved_pose_ncc,
            "diagnostic_dir": path_for_record(diag_dir),
            "notes": self.args.notes or "",
        }
        append_jsonl(Path(self.args.labels_jsonl), record)
        append_csv(Path(self.args.labels_csv), record)
        self.saved_record = record
        print(f"[INFO] Saved manual label: {record['label_id']}")
        print(f"[INFO] JSONL: {self.args.labels_jsonl}")
        print(f"[INFO] CSV:   {self.args.labels_csv}")
        plt.close(self.fig)

    def nudge_slider(self, key: str, delta: float) -> None:
        self.set_slider_value(key, self._slider_value(key) + delta)

    def on_key(self, event) -> None:
        if event.key in ("enter", "return"):
            self.save()
        elif event.key == "left":
            self.nudge_slider("tx", -self.args.keyboard_translation_step)
        elif event.key == "right":
            self.nudge_slider("tx", self.args.keyboard_translation_step)
        elif event.key == "up":
            self.nudge_slider("tz", -self.args.keyboard_translation_step)
        elif event.key == "down":
            self.nudge_slider("tz", self.args.keyboard_translation_step)
        elif event.key == "[":
            self.nudge_slider("ty", -self.args.keyboard_depth_step)
        elif event.key == "]":
            self.nudge_slider("ty", self.args.keyboard_depth_step)
        elif event.key == "w":
            self.nudge_slider("rx", self.args.keyboard_rotation_step)
        elif event.key == "s":
            self.nudge_slider("rx", -self.args.keyboard_rotation_step)
        elif event.key == "q":
            self.nudge_slider("ry", -self.args.keyboard_rotation_step)
        elif event.key == "e":
            self.nudge_slider("ry", self.args.keyboard_rotation_step)
        elif event.key == "a":
            self.nudge_slider("rz", -self.args.keyboard_rotation_step)
        elif event.key == "d":
            self.nudge_slider("rz", self.args.keyboard_rotation_step)

    def run(self) -> dict[str, Any] | None:
        disable_conflicting_matplotlib_keymaps()
        if "agg" in matplotlib.get_backend().lower():
            raise RuntimeError(
                "Interactive placement needs a GUI matplotlib backend. "
                "Do not set MPLBACKEND=Agg for the 'place' command."
            )

        init_pose = pose_from_tensor(self.init_params)
        self.fig = plt.figure(figsize=(13, 9))
        self.ax = self.fig.add_axes([0.05, 0.28, 0.62, 0.68])

        slider_specs = [
            (
                "rx", "rx deg", init_pose["rx"],
                ROTATION_SLIDER_MIN, ROTATION_SLIDER_MAX, 0.1,
            ),
            (
                "ry", "ry deg", init_pose["ry"],
                ROTATION_SLIDER_MIN, ROTATION_SLIDER_MAX, 0.1,
            ),
            (
                "rz", "rz deg", init_pose["rz"],
                ROTATION_SLIDER_MIN, ROTATION_SLIDER_MAX, 0.1,
            ),
            ("tx", "tx mm", init_pose["tx"], -80.0, 80.0, 0.1),
            (
                "ty",
                "ty mm",
                init_pose["ty"],
                self.args.ty_slider_min,
                self.args.ty_slider_max,
                0.1,
            ),
            ("tz", "tz mm", init_pose["tz"], -80.0, 80.0, 0.1),
        ]
        y0 = 0.82
        for index, (key, label, value, vmin, vmax, step) in enumerate(slider_specs):
            vmin, vmax = slider_bounds(value, vmin, vmax)
            y = y0 - index * 0.075
            ax_slider = self.fig.add_axes([0.73, y, 0.16, 0.028])
            ax_text = self.fig.add_axes([0.91, y, 0.06, 0.028])
            self.sliders[key] = Slider(
                ax_slider,
                label,
                valmin=float(vmin),
                valmax=float(vmax),
                valinit=round_pose_value(value),
                valstep=step,
            )
            self.sliders[key].on_changed(
                lambda slider_value, slider_key=key: self.on_slider_changed(
                    slider_key, slider_value
                )
            )
            self.text_boxes[key] = TextBox(
                ax_text,
                "",
                initial=f"{self._slider_value(key):.1f}",
            )
            self.text_boxes[key].on_submit(
                lambda text, slider_key=key: self.on_text_submit(slider_key, text)
            )

        save_ax = self.fig.add_axes([0.73, 0.28, 0.10, 0.055])
        save_button = Button(save_ax, "Save")
        save_button.on_clicked(self.save)

        close_ax = self.fig.add_axes([0.85, 0.28, 0.10, 0.055])
        close_button = Button(close_ax, "Skip")
        close_button.on_clicked(lambda _event: plt.close(self.fig))

        help_text = (
            "Keyboard\n"
            "←/→ tx   ↑/↓ tz\n"
            "[/] depth ty\n"
            "w/s rx   q/e ry   a/d rz\n"
            "Enter = Save"
        )
        self.fig.text(
            0.73,
            0.19,
            help_text,
            fontsize=10,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update()
        plt.show()
        return self.saved_record


def choose_roi(kreg, xray_np: np.ndarray, args: argparse.Namespace, stl_name: str):
    if args.roi is not None:
        return tuple(int(v) for v in args.roi)
    print(f"[INFO] Select ROI for {stl_name}.")
    print("       Use a tight crop around the component/bone this STL represents.")
    return tuple(int(v) for v in kreg.interactive_roi_select(xray_np))


def estimate_initial_params(
    *,
    kreg,
    renderer,
    roi_img: np.ndarray,
    stl_path: Path,
    stl_scale: float,
    stl_center: np.ndarray,
    pixel_spacing: float,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> torch.Tensor:
    base_ty = args.init_ty if args.init_ty is not None else 0.85 * args.sdd
    base = torch.tensor(
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
    if args.init_ty is not None:
        return base

    if args.scale_length_cm is None:
        print(
            "[INFO] No ruler: fitting initial depth directly to the target "
            "segmentation across the full valid depth range"
        )
        try:
            base[4] = float(
                kreg.estimate_initial_ty(
                    renderer,
                    roi_img,
                    base,
                    device,
                    args.sdd,
                    str(out_dir),
                    steps=args.ty_search_steps,
                    implant_percentile=args.implant_percentile,
                    segment_mode=args.segment_mode,
                    stl_path=str(stl_path),
                    stl_scale=stl_scale,
                    stl_center=stl_center,
                )
            )
        except Exception as error:
            fallback_ty = 0.78 * args.sdd
            print(
                f"[WARN] No-ruler depth fitting failed ({error}); "
                f"using fallback ty={fallback_ty:.2f}mm"
            )
            base[4] = fallback_ty
        return base

    try:
        if args.segment_mode == "bone":
            bbox, mask = kreg.segment_bone_bbox(roi_img)
        else:
            bbox, mask = kreg.segment_implant_bbox(
                roi_img, percentile=args.implant_percentile
            )
        target_box = kreg._mask_box(mask)
        if target_box is None:
            target_box = kreg._box_from_bbox(bbox)
        ty = kreg.closed_form_ty(
            str(stl_path),
            stl_scale,
            target_box,
            pixel_spacing,
            args.sdd,
        )
        try:
            ty = kreg.estimate_initial_ty(
                renderer,
                roi_img,
                base,
                device,
                args.sdd,
                str(out_dir),
                steps=args.ty_search_steps,
                ty_min=ty * 0.85,
                ty_max=ty * 1.15,
                implant_percentile=args.implant_percentile,
                segment_mode=args.segment_mode,
                stl_path=str(stl_path),
                stl_scale=stl_scale,
                stl_center=stl_center,
            )
        except Exception as error:
            print(f"[WARN] Narrow depth sweep failed; using closed-form ty ({error})")
        base[4] = float(ty)
    except Exception as error:
        fallback_ty = 0.78 * args.sdd
        print(
            f"[WARN] Automatic depth initialisation failed ({error}); "
            f"using fallback ty={fallback_ty:.2f}mm"
        )
        base[4] = fallback_ty
    return base


def command_place(args: argparse.Namespace) -> None:
    kreg = load_kreg_module()
    xray_path = Path(args.xray)
    xray_np = kreg.load_image_as_array(xray_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scale_points is not None and args.scale_length_cm is None:
        raise ValueError("--scale_points requires --scale_length_cm")
    if args.scale_length_cm is not None:
        if args.scale_length_cm <= 0.0:
            raise ValueError("--scale_length_cm must be positive")
        pixel_spacing = kreg.calibrate_pixel_spacing(
            xray_np,
            args.scale_length_cm,
            str(out_dir / "_scale_calibration"),
            points=args.scale_points,
        )
    else:
        pixel_spacing = float(args.pixel_spacing)
        if pixel_spacing <= 0.0:
            raise ValueError(
                "--pixel_spacing must be positive when no scale ruler is provided"
            )
        print(
            "[INFO] No X-ray ruler calibration: using configured detector "
            f"pixel spacing {pixel_spacing:.6f} mm/pixel"
        )

    device = torch.device(args.device)
    stls = [Path(path) for path in (args.stls or DEFAULT_STLS)]
    names = args.names or [path.stem for path in stls]
    if len(names) != len(stls):
        raise ValueError("--names must have the same number of entries as --stls")

    for stl_index, (stl_path, stl_name) in enumerate(zip(stls, names)):
        roi_xywh = choose_roi(kreg, xray_np, args, stl_name)
        x, y, w, h = roi_xywh
        roi_img = xray_np[y : y + h, x : x + w]
        roi_processed = kreg.preprocess_roi(roi_img)
        target = torch.from_numpy(roi_processed).float().to(device)

        volume, affine, stl_center = kreg.load_and_voxelize(
            str(stl_path),
            voxel_size=args.voxel_size,
            stl_scale=args.stl_scale,
        )
        volume = volume.to(device)
        renderer = kreg.build_renderer(
            volume,
            affine,
            args.sdd,
            pixel_spacing,
            h,
            w,
            device,
            projection_mode="silhouette",
            silhouette_threshold=args.silhouette_threshold,
            silhouette_blur_sigma=args.silhouette_blur_sigma,
        )
        stl_out_dir = out_dir / "_initial_depth" / stl_name
        stl_args = argparse.Namespace(**vars(args))
        initial_rotation = initial_rotation_for_stl(args, stl_index)
        stl_args.init_rx, stl_args.init_ry, stl_args.init_rz = initial_rotation
        common_rotation = tuple(
            float(default if value is None else value)
            for value, default in zip(
                (args.init_rx, args.init_ry, args.init_rz),
                DEFAULT_FIRST_STL_ROTATION,
            )
        )
        if stl_index == 1 and initial_rotation != common_rotation:
            print(
                "[INFO] STL #2 initial rotation override: "
                f"rx={stl_args.init_rx:+.1f}, ry={stl_args.init_ry:+.1f}, "
                f"rz={stl_args.init_rz:+.1f} deg"
            )
        init_params = estimate_initial_params(
            kreg=kreg,
            renderer=renderer,
            roi_img=roi_img,
            stl_path=stl_path,
            stl_scale=args.stl_scale,
            stl_center=stl_center,
            pixel_spacing=pixel_spacing,
            device=device,
            args=stl_args,
            out_dir=stl_out_dir,
        )

        session = ManualPlacementSession(
            kreg=kreg,
            xray_path=xray_path,
            roi_img=roi_img,
            roi_xywh=roi_xywh,
            stl_path=stl_path,
            stl_name=stl_name,
            renderer=renderer,
            stl_center=stl_center,
            target=target,
            init_params=init_params,
            device=device,
            args=stl_args,
            pixel_spacing=pixel_spacing,
            diagnostic_dir=out_dir,
        )
        session.run()


def command_train(args: argparse.Namespace) -> None:
    records = load_labels(Path(args.labels_jsonl))
    if len(records) < 4:
        print(
            "[WARN] You have fewer than 4 labels. The model will train, but it "
            "will mostly memorize. Collect more manual placements for real use."
        )

    image_size = int(args.image_size)
    dataset = PoseLabelDataset(records, image_size=image_size)
    generator = torch.Generator().manual_seed(args.seed)
    if len(dataset) >= 5 and args.val_fraction > 0.0:
        val_count = max(1, int(round(len(dataset) * args.val_fraction)))
        train_count = len(dataset) - val_count
        train_set, val_set = torch.utils.data.random_split(
            dataset,
            [train_count, val_count],
            generator=generator,
        )
    else:
        train_set, val_set = dataset, None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=min(args.batch_size, len(train_set)),
        shuffle=True,
        generator=generator,
    )
    val_loader = (
        torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)
        if val_set is not None and len(val_set) > 0
        else None
    )

    device = torch.device(args.device)
    model = TinyPoseCNN(n_stls=len(dataset.stl_to_id)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for image, stl_id, target in train_loader:
            image = image.to(device)
            stl_id = stl_id.to(device)
            target = target.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(image, stl_id)
            loss = loss_fn(pred, target)
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * image.shape[0]
            count += image.shape[0]
        train_loss = running / max(count, 1)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            values = []
            with torch.no_grad():
                for image, stl_id, target in val_loader:
                    image = image.to(device)
                    stl_id = stl_id.to(device)
                    target = target.to(device)
                    values.append(float(loss_fn(model(image, stl_id), target).item()))
            val_loss = float(np.mean(values))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:04d}/{args.epochs}] "
                f"train={train_loss:.5f} val={val_loss:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stl_to_id": dataset.stl_to_id,
        "target_mean": dataset.target_mean.tolist(),
        "target_std": dataset.target_std.tolist(),
        "pose_keys": list(POSE_KEYS),
        "image_size": image_size,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "labels_jsonl": str(Path(args.labels_jsonl)),
    }
    output_path = Path(args.model_out)
    ensure_parent(output_path)
    torch.save(checkpoint, output_path)
    print(f"[INFO] Saved ML initializer model: {output_path}")


def predict_pose_from_model(
    *,
    model_path: Path,
    xray_path: Path,
    stl_path: Path,
    roi_xywh: tuple[int, int, int, int],
    device: torch.device,
) -> dict[str, float]:
    checkpoint = torch.load(model_path, map_location=device)
    image_size = int(checkpoint["image_size"])
    record = {
        "project_root": str(Path.cwd().resolve()),
        "xray": path_for_record(xray_path),
        "stl": path_for_record(stl_path),
        "stl_name": stl_path.stem,
        "roi_xywh": list(roi_xywh),
        "pose": {key: 0.0 for key in POSE_KEYS},
    }
    dataset = PoseLabelDataset(
        [record],
        image_size=image_size,
        stl_to_id=checkpoint["stl_to_id"],
        target_mean=np.asarray(checkpoint["target_mean"], dtype=np.float32),
        target_std=np.asarray(checkpoint["target_std"], dtype=np.float32),
    )
    model = TinyPoseCNN(n_stls=len(checkpoint["stl_to_id"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image, stl_id, _ = dataset[0]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device), stl_id.unsqueeze(0).to(device))
    pred_np = pred.squeeze(0).cpu().numpy()
    pose_vector = (
        pred_np * np.asarray(checkpoint["target_std"], dtype=np.float32)
        + np.asarray(checkpoint["target_mean"], dtype=np.float32)
    )
    return {key: float(value) for key, value in zip(POSE_KEYS, pose_vector)}


def command_predict(args: argparse.Namespace) -> None:
    kreg = load_kreg_module()
    xray_path = Path(args.xray)
    stl_path = Path(args.stl)
    xray_np = kreg.load_image_as_array(xray_path)
    if args.roi is not None:
        roi_xywh = tuple(int(v) for v in args.roi)
    else:
        roi_xywh = tuple(int(v) for v in kreg.interactive_roi_select(xray_np))
    pose = predict_pose_from_model(
        model_path=Path(args.model),
        xray_path=xray_path,
        stl_path=stl_path,
        roi_xywh=roi_xywh,
        device=torch.device(args.device),
    )
    print("[INFO] Predicted initial pose:")
    for key in POSE_KEYS:
        unit = "deg" if key.startswith("r") else "mm"
        print(f"  {key} = {pose[key]:+.3f} {unit}")

    command = [
        "./.venv/bin/python",
        "-u",
        "KneeRegistration2D3D_V2.py",
        "--stl",
        str(stl_path),
        "--xray",
        str(xray_path),
        "--roi",
        *map(str, roi_xywh),
        "--sdd",
        str(args.sdd),
        "--pixel_spacing",
        str(args.pixel_spacing),
        "--init_rx",
        f"{pose['rx']:.6f}",
        "--init_ry",
        f"{pose['ry']:.6f}",
        "--init_rz",
        f"{pose['rz']:.6f}",
        "--init_tx",
        f"{pose['tx']:.6f}",
        "--init_ty",
        f"{pose['ty']:.6f}",
        "--init_tz",
        f"{pose['tz']:.6f}",
        "--chamfer_refine",
        "--warm_start_passes",
        str(args.warm_start_passes),
        "--out_dir",
        args.out_dir,
    ]
    print("\n[INFO] V2 command using ML initialisation:")
    print(" ".join(command))
    if args.run_v2:
        subprocess.run(command, check=True)


def add_common_placement_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--xray", required=True, help="X-ray image path")
    parser.add_argument(
        "--stls",
        nargs="+",
        default=list(DEFAULT_STLS),
        help="STL files to place one at a time",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Optional human-readable STL names; same length as --stls",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Shared ROI. If omitted, select an ROI interactively for each STL.",
    )
    parser.add_argument("--pixel_spacing", type=float, default=0.194)
    parser.add_argument("--scale_length_cm", type=float, default=None)
    parser.add_argument("--scale_points", type=float, nargs=4, default=None)
    parser.add_argument("--sdd", type=float, default=1024.0)
    parser.add_argument("--voxel_size", type=float, default=0.5)
    parser.add_argument("--stl_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--segment_mode", choices=("implant", "bone"), default="implant")
    parser.add_argument("--implant_percentile", type=float, default=80.0)
    parser.add_argument("--silhouette_threshold", type=float, default=0.01)
    parser.add_argument("--silhouette_blur_sigma", type=float, default=1.5)
    parser.add_argument("--ty_search_steps", type=int, default=13)
    parser.add_argument("--init_rx", type=float, default=None,
                        help="Common initial rx in degrees (default: 0)")
    parser.add_argument("--init_ry", type=float, default=None,
                        help="Common initial ry in degrees (default: 0)")
    parser.add_argument("--init_rz", type=float, default=None,
                        help="Common initial rz in degrees (default: 0)")
    parser.add_argument("--init_tx", type=float, default=0.0)
    parser.add_argument("--init_ty", type=float, default=None)
    parser.add_argument("--init_tz", type=float, default=0.0)
    parser.add_argument(
        "--second_init_rx",
        type=float,
        default=None,
        help=("Optional second-STL rx override. With no explicit common or "
              "second-STL rotation, the historical default is -90 degrees."),
    )
    parser.add_argument(
        "--second_init_ry",
        type=float,
        default=None,
        help="Optional second-STL ry override (otherwise inherit common ry)",
    )
    parser.add_argument(
        "--second_init_rz",
        type=float,
        default=None,
        help="Optional second-STL rz override (otherwise inherit common rz)",
    )
    parser.add_argument(
        "--disable_second_stl_init_override",
        action="store_true",
        help="Use the normal --init_rx/--init_ry/--init_rz values for every STL",
    )
    parser.add_argument("--ty_slider_min", type=float, default=600.0)
    parser.add_argument("--ty_slider_max", type=float, default=950.0)
    parser.add_argument("--keyboard_translation_step", type=float, default=0.1)
    parser.add_argument("--keyboard_depth_step", type=float, default=0.1)
    parser.add_argument("--keyboard_rotation_step", type=float, default=0.1)
    parser.add_argument("--labels_jsonl", default="training/manual_pose_labels.jsonl")
    parser.add_argument("--labels_csv", default="training/manual_pose_labels.csv")
    parser.add_argument("--out_dir", default="training/manual_placements")
    parser.add_argument("--notes", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual placement and ML initialisation for 2D-3D registration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    place = subparsers.add_parser(
        "place",
        help="Interactively place one or more STLs on an X-ray and save labels",
    )
    add_common_placement_args(place)
    place.set_defaults(func=command_place)

    train = subparsers.add_parser(
        "train",
        help="Train a small pose initializer from manual labels",
    )
    train.add_argument("--labels_jsonl", default="training/manual_pose_labels.jsonl")
    train.add_argument("--model_out", default="training/ml_init_pose_model.pt")
    train.add_argument("--image_size", type=int, default=96)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--batch_size", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--val_fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="cpu")
    train.add_argument("--print_every", type=int, default=25)
    train.set_defaults(func=command_train)

    predict = subparsers.add_parser(
        "predict",
        help="Predict an initial pose and print a KneeRegistration2D3D_V2 command",
    )
    predict.add_argument("--model", default="training/ml_init_pose_model.pt")
    predict.add_argument("--xray", required=True)
    predict.add_argument("--stl", required=True)
    predict.add_argument("--roi", type=int, nargs=4, default=None)
    predict.add_argument("--pixel_spacing", type=float, default=0.194)
    predict.add_argument("--sdd", type=float, default=1024.0)
    predict.add_argument("--warm_start_passes", type=int, default=3)
    predict.add_argument("--out_dir", default="results/ml_init_prediction")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--run_v2", action="store_true")
    predict.set_defaults(func=command_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
