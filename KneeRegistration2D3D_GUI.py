#!/usr/bin/env python3
"""
Graphical launcher for KneeRegistration2D3D_V2.py.

This GUI does not reimplement the registration pipeline. It collects the
analysis settings, optionally asks KneeRegistration2D3D_MLInit.py for an ML
initial pose, and then launches the existing core script with the selected
arguments. The core script still owns the ROI selector, scale-line selector,
initial-position preview, optimisation, and saved outputs.
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox
from tkinter import scrolledtext
from tkinter import ttk


ROOT = Path(__file__).resolve().parent
V2_SCRIPT = ROOT / "KneeRegistration2D3D_V2.py"
ML_SCRIPT = ROOT / "KneeRegistration2D3D_MLInit.py"


POSE_RE = re.compile(
    r"^\s*(rx|ry|rz|tx|ty|tz)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)
ROI_RE = re.compile(r"--roi\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")


def default_python() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def display_path(path: str) -> str:
    if not path:
        return path
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path))


def split_values(value: str, expected: int, label: str) -> list[str]:
    cleaned = value.replace(",", " ").strip()
    if not cleaned:
        return []
    parts = cleaned.split()
    if len(parts) != expected:
        raise ValueError(f"{label} must contain {expected} numbers")
    return parts


class RegistrationGui(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("2D-3D Knee Registration Launcher")
        self.geometry("1040x760")
        self.minsize(900, 620)

        self.vars: dict[str, StringVar | BooleanVar] = {}
        self.process: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None
        self.stop_requested = False
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_variables()
        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_variables(self) -> None:
        string_defaults = {
            "python": default_python(),
            "mplconfigdir": "/private/tmp/matplotlib",
            "stl": "3D_files/Phase3FemurM.stl",
            "xray": "Xray/ASIP1.jpg",
            "out_dir": "results/ASIP1",
            "ml_model": "training/ml_init_pose_model.pt",
            "ml_device": "cpu",
            "roi": "",
            "scale_length_cm": "16",
            "scale_points": "",
            "pixel_spacing": "0.194",
            "stl_scale": "1.0",
            "sdd": "1024",
            "voxel_size": "0.5",
            "projection_mode": "silhouette",
            "silhouette_threshold": "0.01",
            "silhouette_blur_sigma": "1.5",
            "init_rx": "0",
            "init_ry": "0",
            "init_rz": "0",
            "final_rx_offset_deg": "0",
            "init_tx": "0",
            "init_ty": "",
            "init_tz": "0",
            "ty_search_steps": "13",
            "ty_search_min": "",
            "ty_search_max": "",
            "implant_percentile": "60",
            "coarse_iters": "100",
            "fine_iters": "300",
            "lr": "0.005",
            "translation_lr": "0.5",
            "grid_angles": "5",
            "grid_range": "30.0",
            "device": "auto",
            "edge_ncc_weight": "0.6",
            "rx_polish_range": "5.0",
            "rx_polish_step": "0.1",
            "chamfer_ty_range": "20.0",
            "chamfer_ty_step": "1.0",
            "chamfer_rx_range": "5.0",
            "chamfer_rx_step": "1.0",
            "chamfer_segment_mode": "implant",
            "chamfer_iou_weight": "1.0",
            "chamfer_outside_weight": "1.0",
            "segment_mode": "implant",
            "warm_start_passes": "0",
            "warm_start_min_delta": "0.0001",
            "ty_guard_mm": "100",
        }
        bool_defaults = {
            "use_ml_init": False,
            "grid_search": False,
            "no_initial_preview": False,
            "preview_only": False,
            "chamfer_refine": True,
            "chamfer_combined_objective": True,
            "warm_start_free_ty": False,
        }
        for name, value in string_defaults.items():
            self.vars[name] = StringVar(value=value)
        for name, value in bool_defaults.items():
            self.vars[name] = BooleanVar(value=value)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=(10, 6))

        files_tab = ttk.Frame(notebook, padding=10)
        geometry_tab = ttk.Frame(notebook, padding=10)
        pose_tab = ttk.Frame(notebook, padding=10)
        opt_tab = ttk.Frame(notebook, padding=10)
        chamfer_tab = ttk.Frame(notebook, padding=10)
        command_tab = ttk.Frame(notebook, padding=10)

        notebook.add(files_tab, text="Files")
        notebook.add(geometry_tab, text="Geometry")
        notebook.add(pose_tab, text="Initial Pose")
        notebook.add(opt_tab, text="Optimisation")
        notebook.add(chamfer_tab, text="Chamfer/Segmentation")
        notebook.add(command_tab, text="Command Log")

        self._build_files_tab(files_tab)
        self._build_geometry_tab(geometry_tab)
        self._build_pose_tab(pose_tab)
        self._build_optimisation_tab(opt_tab)
        self._build_chamfer_tab(chamfer_tab)
        self._build_command_tab(command_tab)

        action_bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        action_bar.pack(fill="x")
        self.run_button = ttk.Button(
            action_bar, text="Run Registration", command=self.start_run
        )
        self.run_button.pack(side="left")
        self.preview_button = ttk.Button(
            action_bar, text="Show Command", command=self.show_command
        )
        self.preview_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(
            action_bar, text="Stop", command=self.stop_run, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Label(
            action_bar,
            text=(
                "ROI, scale-line adjustment, and initial-position approval "
                "open in the existing core windows."
            ),
        ).pack(side="left", padx=(16, 0))

    def _label(self, parent: ttk.Frame, row: int, text: str) -> None:
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=4)

    def _entry(
        self,
        parent: ttk.Frame,
        row: int,
        name: str,
        label: str,
        width: int = 28,
        column: int = 1,
    ) -> ttk.Entry:
        self._label(parent, row, label)
        entry = ttk.Entry(parent, textvariable=self.vars[name], width=width)
        entry.grid(row=row, column=column, sticky="ew", pady=4, padx=(8, 0))
        parent.columnconfigure(column, weight=1)
        return entry

    def _choice(
        self,
        parent: ttk.Frame,
        row: int,
        name: str,
        label: str,
        values: tuple[str, ...],
    ) -> ttk.Combobox:
        self._label(parent, row, label)
        combo = ttk.Combobox(
            parent,
            textvariable=self.vars[name],
            values=values,
            state="readonly",
            width=26,
        )
        combo.grid(row=row, column=1, sticky="w", pady=4, padx=(8, 0))
        return combo

    def _check(
        self,
        parent: ttk.Frame,
        row: int,
        name: str,
        label: str,
        columnspan: int = 2,
    ) -> ttk.Checkbutton:
        check = ttk.Checkbutton(parent, text=label, variable=self.vars[name])
        check.grid(row=row, column=0, columnspan=columnspan, sticky="w", pady=4)
        return check

    def _browse_button(
        self,
        parent: ttk.Frame,
        row: int,
        name: str,
        kind: str,
        filetypes: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        command = lambda: self._browse(name, kind, filetypes)
        ttk.Button(parent, text="Browse", command=command).grid(
            row=row, column=2, sticky="w", padx=(8, 0)
        )

    def _browse(
        self,
        name: str,
        kind: str,
        filetypes: tuple[tuple[str, str], ...] | None,
    ) -> None:
        current = str(self.vars[name].get()).strip()
        initial = ROOT / current if current and not Path(current).is_absolute() else ROOT
        if kind == "directory":
            selected = filedialog.askdirectory(initialdir=str(initial.parent))
        else:
            selected = filedialog.askopenfilename(
                initialdir=str(initial.parent),
                filetypes=filetypes or (("All files", "*.*"),),
            )
        if selected:
            self.vars[name].set(display_path(selected))

    def _build_files_tab(self, parent: ttk.Frame) -> None:
        self._entry(parent, 0, "python", "Python executable", width=60)
        self._browse_button(
            parent,
            0,
            "python",
            "file",
            (("Python", "python*"), ("All files", "*.*")),
        )
        self._entry(parent, 1, "mplconfigdir", "MPLCONFIGDIR", width=60)
        self._entry(parent, 2, "stl", "STL file", width=60)
        self._browse_button(
            parent,
            2,
            "stl",
            "file",
            (("STL files", "*.stl"), ("All files", "*.*")),
        )
        self._entry(parent, 3, "xray", "X-ray image", width=60)
        self._browse_button(
            parent,
            3,
            "xray",
            "file",
            (
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
        self._entry(parent, 4, "out_dir", "Output directory", width=60)
        self._browse_button(parent, 4, "out_dir", "directory")
        self._check(
            parent,
            5,
            "use_ml_init",
            "Use trained ML initializer from KneeRegistration2D3D_MLInit.py",
        )
        self._entry(parent, 6, "ml_model", "ML model path", width=60)
        self._browse_button(
            parent,
            6,
            "ml_model",
            "file",
            (("PyTorch model", "*.pt *.pth"), ("All files", "*.*")),
        )
        self._entry(parent, 7, "ml_device", "ML device", width=18)
        ttk.Label(
            parent,
            text=(
                "If ML is enabled and ROI is blank, the ML step opens the ROI "
                "selector first and reuses that ROI for the main run."
            ),
            foreground="#444",
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _build_geometry_tab(self, parent: ttk.Frame) -> None:
        self._entry(parent, 0, "roi", "ROI x y w h (optional)", width=38)
        self._entry(parent, 1, "scale_length_cm", "Scale length cm", width=18)
        self._entry(parent, 2, "scale_points", "Scale points x1 y1 x2 y2", width=38)
        self._entry(parent, 3, "pixel_spacing", "Pixel spacing mm", width=18)
        self._entry(parent, 4, "stl_scale", "STL scale", width=18)
        self._entry(parent, 5, "sdd", "Source-detector distance mm", width=18)
        self._entry(parent, 6, "voxel_size", "Voxel size mm", width=18)
        self._choice(
            parent,
            7,
            "projection_mode",
            "Projection mode",
            ("silhouette", "attenuation"),
        )
        self._entry(parent, 8, "silhouette_threshold", "Silhouette threshold")
        self._entry(parent, 9, "silhouette_blur_sigma", "Silhouette blur sigma")

    def _build_pose_tab(self, parent: ttk.Frame) -> None:
        self._entry(parent, 0, "init_rx", "Initial rx deg")
        self._entry(parent, 1, "init_ry", "Initial ry deg")
        self._entry(parent, 2, "init_rz", "Initial rz deg")
        self._entry(parent, 3, "final_rx_offset_deg", "Final rx offset deg")
        self._entry(parent, 4, "init_tx", "Initial tx mm")
        self._entry(parent, 5, "init_ty", "Initial ty mm (blank = auto)")
        self._entry(parent, 6, "init_tz", "Initial tz mm")
        self._check(parent, 7, "no_initial_preview", "Skip initial preview")
        self._check(parent, 8, "preview_only", "Preview only, then exit")

    def _build_optimisation_tab(self, parent: ttk.Frame) -> None:
        self._entry(parent, 0, "coarse_iters", "Coarse iterations")
        self._entry(parent, 1, "fine_iters", "Fine iterations")
        self._entry(parent, 2, "lr", "Rotation learning rate")
        self._entry(parent, 3, "translation_lr", "Translation learning rate")
        self._entry(parent, 4, "edge_ncc_weight", "Edge NCC weight")
        self._entry(parent, 5, "rx_polish_range", "rx polish range deg")
        self._entry(parent, 6, "rx_polish_step", "rx polish step deg")
        self._entry(parent, 7, "warm_start_passes", "Warm-start passes")
        self._entry(parent, 8, "warm_start_min_delta", "Warm-start min delta")
        self._check(parent, 9, "warm_start_free_ty", "Allow warm starts to optimise ty")
        self._entry(parent, 10, "ty_guard_mm", "ty guard mm")
        self._entry(parent, 11, "ty_search_steps", "ty search steps")
        self._entry(parent, 12, "ty_search_min", "ty search min mm")
        self._entry(parent, 13, "ty_search_max", "ty search max mm")
        self._choice(parent, 14, "device", "Device", ("auto", "cpu", "cuda"))
        self._check(parent, 15, "grid_search", "Run coarse grid search")
        self._entry(parent, 16, "grid_angles", "Grid angle count")
        self._entry(parent, 17, "grid_range", "Grid range deg")

    def _build_chamfer_tab(self, parent: ttk.Frame) -> None:
        self._check(parent, 0, "chamfer_refine", "Enable Chamfer refinement")
        self._choice(
            parent,
            1,
            "chamfer_segment_mode",
            "Chamfer segmentation",
            ("implant", "bone"),
        )
        self._check(
            parent,
            2,
            "chamfer_combined_objective",
            "Use combined Chamfer objective",
        )
        self._entry(parent, 3, "chamfer_iou_weight", "Chamfer IoU weight")
        self._entry(parent, 4, "chamfer_outside_weight", "Chamfer outside weight")
        self._entry(parent, 5, "chamfer_ty_range", "Chamfer ty range mm")
        self._entry(parent, 6, "chamfer_ty_step", "Chamfer ty step mm")
        self._entry(parent, 7, "chamfer_rx_range", "Chamfer rx range deg")
        self._entry(parent, 8, "chamfer_rx_step", "Chamfer rx step deg")
        self._choice(parent, 9, "segment_mode", "Depth segmentation", ("implant", "bone"))
        self._entry(parent, 10, "implant_percentile", "Implant percentile")

    def _build_command_tab(self, parent: ttk.Frame) -> None:
        self.output = scrolledtext.ScrolledText(
            parent,
            wrap="word",
            height=28,
            font=("Menlo", 11),
        )
        self.output.pack(fill="both", expand=True)

    def get(self, name: str) -> str:
        return str(self.vars[name].get()).strip()

    def get_bool(self, name: str) -> bool:
        return bool(self.vars[name].get())

    def _add_value(
        self,
        cmd: list[str],
        flag: str,
        name: str,
        *,
        required: bool = False,
        override: str | None = None,
    ) -> None:
        value = override if override is not None else self.get(name)
        if not value:
            if required:
                raise ValueError(f"{name} is required")
            return
        cmd.extend([flag, value])

    def _build_v2_command(self, overrides: dict[str, str | list[str]] | None = None) -> list[str]:
        overrides = overrides or {}
        python = self.get("python") or default_python()
        cmd = [python, "-u", str(V2_SCRIPT)]
        self._add_value(cmd, "--stl", "stl", required=True)
        self._add_value(cmd, "--xray", "xray", required=True)

        roi_override = overrides.get("roi")
        roi_values = (
            list(roi_override)
            if isinstance(roi_override, list)
            else split_values(self.get("roi"), 4, "ROI")
        )
        if roi_values:
            cmd.extend(["--roi", *roi_values])

        scalar_args = [
            ("--voxel_size", "voxel_size"),
            ("--stl_scale", "stl_scale"),
            ("--sdd", "sdd"),
            ("--pixel_spacing", "pixel_spacing"),
            ("--scale_length_cm", "scale_length_cm"),
            ("--projection_mode", "projection_mode"),
            ("--silhouette_threshold", "silhouette_threshold"),
            ("--silhouette_blur_sigma", "silhouette_blur_sigma"),
            ("--init_rx", "init_rx"),
            ("--init_ry", "init_ry"),
            ("--init_rz", "init_rz"),
            ("--final_rx_offset_deg", "final_rx_offset_deg"),
            ("--init_tx", "init_tx"),
            ("--init_ty", "init_ty"),
            ("--init_tz", "init_tz"),
            ("--ty_search_steps", "ty_search_steps"),
            ("--ty_search_min", "ty_search_min"),
            ("--ty_search_max", "ty_search_max"),
            ("--implant_percentile", "implant_percentile"),
            ("--coarse_iters", "coarse_iters"),
            ("--fine_iters", "fine_iters"),
            ("--lr", "lr"),
            ("--translation_lr", "translation_lr"),
            ("--grid_angles", "grid_angles"),
            ("--grid_range", "grid_range"),
            ("--device", "device"),
            ("--out_dir", "out_dir"),
            ("--edge_ncc_weight", "edge_ncc_weight"),
            ("--rx_polish_range", "rx_polish_range"),
            ("--rx_polish_step", "rx_polish_step"),
            ("--chamfer_ty_range", "chamfer_ty_range"),
            ("--chamfer_ty_step", "chamfer_ty_step"),
            ("--chamfer_rx_range", "chamfer_rx_range"),
            ("--chamfer_rx_step", "chamfer_rx_step"),
            ("--chamfer_segment_mode", "chamfer_segment_mode"),
            ("--chamfer_iou_weight", "chamfer_iou_weight"),
            ("--chamfer_outside_weight", "chamfer_outside_weight"),
            ("--segment_mode", "segment_mode"),
            ("--warm_start_passes", "warm_start_passes"),
            ("--warm_start_min_delta", "warm_start_min_delta"),
            ("--ty_guard_mm", "ty_guard_mm"),
        ]
        for flag, name in scalar_args:
            override = overrides.get(name)
            self._add_value(
                cmd,
                flag,
                name,
                override=override if isinstance(override, str) else None,
            )

        scale_points = split_values(self.get("scale_points"), 4, "Scale points")
        if scale_points:
            cmd.extend(["--scale_points", *scale_points])

        bool_args = [
            ("--grid_search", "grid_search"),
            ("--no_initial_preview", "no_initial_preview"),
            ("--preview_only", "preview_only"),
            ("--chamfer_refine", "chamfer_refine"),
            ("--chamfer_combined_objective", "chamfer_combined_objective"),
            ("--warm_start_free_ty", "warm_start_free_ty"),
        ]
        for flag, name in bool_args:
            if self.get_bool(name):
                cmd.append(flag)
        return cmd

    def _build_ml_command(self) -> list[str]:
        model = self.get("ml_model")
        if not model:
            raise ValueError("ML model path is required when ML initializer is enabled")
        python = self.get("python") or default_python()
        cmd = [
            python,
            "-u",
            str(ML_SCRIPT),
            "predict",
            "--model",
            model,
            "--xray",
            self.get("xray"),
            "--stl",
            self.get("stl"),
            "--pixel_spacing",
            self.get("pixel_spacing"),
            "--sdd",
            self.get("sdd"),
            "--warm_start_passes",
            self.get("warm_start_passes"),
            "--out_dir",
            self.get("out_dir"),
            "--device",
            self.get("ml_device") or "cpu",
        ]
        roi_values = split_values(self.get("roi"), 4, "ROI")
        if roi_values:
            cmd.extend(["--roi", *roi_values])
        return cmd

    def _make_env(self) -> dict[str, str]:
        env = os.environ.copy()
        mplconfigdir = self.get("mplconfigdir")
        if mplconfigdir:
            Path(mplconfigdir).mkdir(parents=True, exist_ok=True)
            env["MPLCONFIGDIR"] = mplconfigdir
        return env

    def show_command(self) -> None:
        try:
            self.output.delete("1.0", "end")
            if self.get_bool("use_ml_init"):
                self._log_line("ML initializer command:")
                self._log_line(shlex.join(self._build_ml_command()))
                self._log_line("")
            self._log_line("Registration command:")
            self._log_line(shlex.join(self._build_v2_command()))
        except Exception as error:
            messagebox.showerror("Invalid settings", str(error))

    def start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.output.delete("1.0", "end")
        self.stop_requested = False
        self.run_button.configure(state="disabled")
        self.preview_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self.worker.start()

    def stop_run(self) -> None:
        self.stop_requested = True
        if self.process and self.process.poll() is None:
            self._queue_log("[GUI] Stopping current process...\n")
            self.process.terminate()

    def _run_pipeline(self) -> None:
        try:
            env = self._make_env()
            overrides: dict[str, str | list[str]] = {}

            if self.get_bool("use_ml_init"):
                ml_cmd = self._build_ml_command()
                self._queue_log("[GUI] Running ML initializer first.\n")
                ml_output, ml_code = self._run_command(ml_cmd, env)
                if ml_code != 0:
                    self._queue_log("[GUI] ML initializer failed; registration not started.\n")
                    return
                pose = self._parse_ml_pose(ml_output)
                if pose:
                    for name, value in pose.items():
                        text = f"{value:.6f}"
                        overrides[name] = text
                        self.after(0, self.vars[name].set, text)
                    self._queue_log("[GUI] ML initial pose applied.\n")
                else:
                    self._queue_log(
                        "[GUI] Warning: no ML pose was parsed; using GUI initial pose.\n"
                    )
                if not self.get("roi"):
                    roi = self._parse_roi(ml_output)
                    if roi:
                        overrides["roi"] = roi
                        self.after(0, self.vars["roi"].set, " ".join(roi))
                        self._queue_log("[GUI] ROI selected during ML step will be reused.\n")

            if self.stop_requested:
                return
            v2_cmd = self._build_v2_command(overrides)
            self._queue_log("[GUI] Running registration.\n")
            _output, _code = self._run_command(v2_cmd, env)
        except Exception as error:
            self._queue_log(f"[GUI] Error: {error}\n")
            self.after(0, messagebox.showerror, "Registration launcher", str(error))
        finally:
            self.process = None
            self.after(0, self._finish_run)

    def _run_command(self, cmd: list[str], env: dict[str, str]) -> tuple[str, int]:
        self._queue_log("$ " + shlex.join(cmd) + "\n")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=None,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        assert self.process.stdout is not None
        for line in self.process.stdout:
            lines.append(line)
            self._queue_log(line)
        code = self.process.wait()
        self._queue_log(f"[GUI] Process exited with code {code}\n\n")
        return "".join(lines), int(code)

    def _parse_ml_pose(self, output: str) -> dict[str, float]:
        pose: dict[str, float] = {}
        for line in output.splitlines():
            match = POSE_RE.match(line)
            if match:
                pose[match.group(1)] = float(match.group(2))
        return pose

    def _parse_roi(self, output: str) -> list[str]:
        match = ROI_RE.search(output)
        if not match:
            return []
        return list(match.groups())

    def _finish_run(self) -> None:
        self.run_button.configure(state="normal")
        self.preview_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _queue_log(self, text: str) -> None:
        self.log_queue.put(text)

    def _log_line(self, text: str) -> None:
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.output.insert("end", text)
                self.output.see("end")
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)


def launch_gui() -> None:
    app = RegistrationGui()
    app.mainloop()


def main() -> None:
    launch_gui()


if __name__ == "__main__":
    main()
