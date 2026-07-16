# Knee 2D–3D Registration

This project was developed by **Alexandre Bédard** between **July 1st and July 5th, 2026**. For a side project during his stay at NDORMS

This is experimental research code. Some coding errors, numerical issues, or edge cases may still be present. The goal of this repository is to test a Python pipeline for registering a 3D knee STL model onto a 2D X-ray image.

---

## What the code does

The script `KneeRegistration2D3D_V2.py` performs a 2D–3D registration between a 3D STL model and a 2D radiograph.

The pipeline:

1. Loads a knee STL file using `trimesh`.
2. Voxelizes the STL into a 3D density volume.
3. Converts the volume into a `torch` tensor for use with DiffDRR.
4. Loads and crops the X-ray image to the selected region of interest.
5. Normalizes the radiograph.
6. Sets up a perspective camera model using:

   * source-to-detector distance,
   * pixel spacing,
   * detector dimensions matching the cropped ROI.
7. Runs a multi-resolution optimization from coarse resolution to full resolution.
8. Uses Adam optimization with a learning-rate schedule.
9. Minimizes a loss based on `1 - NCC`, where NCC is the normalized cross-correlation between the synthetic DRR and the X-ray ROI.
10. Optionally performs extra refinement using edge-based NCC and Chamfer distance.
11. Outputs the final 6-DOF pose of the STL relative to the X-ray.

The final pose includes:

* Euler rotations in degrees:

  * `rx`
  * `ry`
  * `rz`
* Translations in millimetres:

  * `tx`
  * `ty`
  * `tz`

---

## Required folder structure

Before running the code, place your files in the following folders:

```text
project_folder/
│
├── KneeRegistration2D3D_V2.py
│
├── 3D_files/
│   └── Your_Beautiful_3Dmodel.stl
│
├── Xray/
│   └── The_XRay_Of_Someone.jpg
│
└── results/
```

Put your 3D STL files inside:

```text
3D_files/
```

Put your X-ray images inside:

```text
Xray/
```

The results will be saved in the folder specified by `--out_dir`.

---

## Example command

In your terminal, run:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib ./.venv/bin/python -u KneeRegistration2D3D_V2.py \
  --stl "3D_files/Your_Beautiful_3Dmodel.stl" \
  --xray "Xray/The_XRay_Of_Someone.jpg" \
  --scale_length_cm 16 \
  --stl_scale 1.0 \
  --sdd 1024 \
  --voxel_size 0.5 \
  --warm_start_passes 0 \
  --init_rx 0 \
  --init_ry 0 \
  --init_rz 0 \
  --edge_ncc_weight 0.6 \
  --rx_polish_range 5.0 \
  --rx_polish_step 0.1 \
  --chamfer_refine \
  --chamfer_segment_mode implant \
  --chamfer_combined_objective \
  --chamfer_iou_weight 1.0 \
  --chamfer_outside_weight 1.0 \
  --chamfer_ty_range 20.0 \
  --chamfer_ty_step 1.0 \
  --chamfer_rx_range 5.0 \
  --chamfer_rx_step 1.0 \
  --ty_guard_mm 100 \
  --translation_lr 0.5 \
  --coarse_iters 100 \
  --fine_iters 300 \
  --segment_mode implant \
  --implant_percentile 60 \
  --out_dir "results/name_of_your_run"
```

The `MPLCONFIGDIR=/private/tmp/matplotlib` part is useful on macOS to avoid permission errors with Matplotlib.

---

## Main input arguments

### File inputs

| Argument    | Description                                  |
| ----------- | -------------------------------------------- |
| `--stl`     | Path to the 3D STL model.                    |
| `--xray`    | Path to the 2D X-ray image.                  |
| `--out_dir` | Folder where the output files will be saved. |

---

### Geometry and camera parameters

| Argument            | Description                                                                                                                                                                                               |
| ------------------- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--scale_length_cm` | Approximate real-world size used to scale the X-ray or object. This helps match the physical size of the STL and radiograph. Look for the scale on the X-Ray before and change this variable accordingly. |
| `--stl_scale`       | Scaling factor applied directly to the STL. Use this if the STL is too large or too small compared with the X-ray. Usualy, always 1.                                                                      |
| `--sdd`             | Source-to-detector distance in millimetres. This defines the perspective projection geometry. Keep 1024 if you have no clue.                                                                              |
| `--voxel_size`      | Size of each voxel used when converting the STL into a 3D volume. Smaller values give more detail but increase computation time. Good compromise is 0.5.                                                  |

---

### Initial pose, helpful for optimization

| Argument        | Description                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--init_rx`     | Initial rotation around the X axis, in degrees.                                                                             |
| `--init_ry`     | Initial rotation around the Y axis, in degrees.                                                                             |
| `--init_rz`     | Initial rotation around the Z axis, in degrees.                                                                             |
| `--ty_guard_mm` | Maximum allowed search or correction range for translation in the Y direction. This helps prevent unrealistic translations. |

A good initial pose is important because 2D–3D registration can get stuck in local minima.

---

### Optimization parameters

| Argument              | Description                                                                                              |
| --------------------- |----------------------------------------------------------------------------------------------------------|
| `--coarse_iters`      | Number of iterations for the coarse optimization stage. I recommand 300 ish                              |
| `--fine_iters`        | Number of iterations for the fine optimization stage. I recommand 500 or even 1000 (longer tho)          |
| `--translation_lr`    | Learning rate for translation updates. A smaller value gives slower but more stable translation changes. |
| `--warm_start_passes` | Number of warm-start passes before the main optimization. Use `0` to disable.                            |

---

### NCC and edge-based matching

| Argument               | Description                                                                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--edge_ncc_weight`    | Weight given to the edge-based NCC term. Higher values make the optimizer focus more on bone or implant edges rather than image intensity alone.      |
| `--segment_mode`       | Defines how the X-ray is segmented before matching. For example, `implant` focuses on implant-like high-intensity structures.                         |
| `--implant_percentile` | Percentile threshold used to isolate bright implant-like regions. Lower values include more pixels; higher values keep only the brightest structures. |

---

### Rotation polish

| Argument            | Description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| `--rx_polish_range` | Search range around the final X rotation, in degrees.                                   |
| `--rx_polish_step`  | Step size used during the X rotation polish. Smaller steps are more precise but slower. |

The rotation polish is useful when the optimizer is close to the correct solution but still needs a small correction in `rx`.

---

### Chamfer refinement

| Argument                       | Description                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| `--chamfer_refine`             | Enables the Chamfer refinement step.                                                    |
| `--chamfer_segment_mode`       | Defines the segmentation mode used for the Chamfer objective.                           |
| `--chamfer_combined_objective` | Combines Chamfer distance with other metrics, such as IoU or outside-mask penalties.    |
| `--chamfer_iou_weight`         | Weight of the intersection-over-union term in the Chamfer refinement.                   |
| `--chamfer_outside_weight`     | Weight penalizing projected STL edges that fall outside the target segmented structure. |
| `--chamfer_ty_range`           | Search range for translation in the Y direction during Chamfer refinement.              |
| `--chamfer_ty_step`            | Step size for Y translation during Chamfer refinement.                                  |
| `--chamfer_rx_range`           | Search range for X rotation during Chamfer refinement.                                  |
| `--chamfer_rx_step`            | Step size for X rotation during Chamfer refinement.                                     |

Chamfer refinement is useful when the silhouette or edge alignment is more important than the intensity-based NCC score.

---

## Outputs

The code saves the following outputs in the selected result folder:

```text
results/name_of_your_run/
```

Typical outputs include:

1. A final 6-DOF pose.
2. A JSON file containing the final pose.
3. The final NCC score.
4. A 3-panel comparison figure:

   * X-ray ROI,
   * best synthetic DRR,
   * overlay between the X-ray and DRR.
5. Additional diagnostic images depending on the selected options.

---

## Notes on interpretation

A higher NCC score generally means better image similarity between the X-ray and the generated DRR. However, NCC alone does not always guarantee perfect anatomical alignment.

For example, the NCC score can improve even when the STL is slightly misplaced if the global image intensity looks similar. This is why edge-based NCC, Chamfer refinement, and visual inspection of the overlay are important.

The final alignment should always be checked visually.

---

## Current development status

`KneeRegistration2D3D_V2.py` is still under active development.

Some parts of the code are experimental, including the data-generation and training-related functions that may later be used to improve or initialize the 2D–3D registration pipeline.

Future improvements may include:

* better automatic initialization,
* improved segmentation of the X-ray,
* more robust STL scaling,
* faster Chamfer refinement,
* better handling of local minima,
* automatic detection of failed registrations,
* training data generation for future machine-learning-based initialization.

---

## Disclaimer

This code is intended for research and development only. It has not been clinically validated and should not be used for patient care or surgical planning without proper validation.

This code has been made with love and care from Alexandre Bédard. For all my trials, go see the pdf "SideProjects_2026.pdf" in the root of this repository. It contains all my trials and errors, and the lessons I learned from them.
