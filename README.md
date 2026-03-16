# RGMC Cloud Robotics 2026

This repository provides the models and IOU evaluator for the **Cloud Robotics** track at the 11th Robotic Grasping and Manipulation Competition (RGMC) during the **2026 IEEE International Conference on Robotics & Automation (ICRA)** in Vienna, Austria.

## Overview

Cloud Robotics track teams are challenged to develop robust manipulation solutions using remote access to the cloud robotics platform [CloudGripper](https://cloudgripper.org) at KTH Royal Institute of Technology.

The API to interact with the robot is available at: https://github.com/cloudgripper/cloudgripper-api

This track focuses on **generalization and robustness** properties of the methods developed. The teams will work on two foundational manipulation tasks:

1. **Planar Pushing** — pushing objects to target locations
2. **Linear Deformable Object Shape Control** — controlling the shape of linear deformable objects like ropes

---

## Repository Contents

```
.
├── config/
│   └── base-camera-calibration/   # Per-robot fisheye camera intrinsics (YAML)
├── models/                        # STL files for Task 1 objects
├── sample_images/                 # Example base-camera images for testing
├── src/
│   └── rgmc_cloud_robotics_2026/  # Installable Python package
│       ├── __init__.py
│       ├── shapes.py              # 3D corner definitions for Task 1 objects (mm)
│       ├── calib_fisheye_cam.py   # Fisheye camera calibration utilities
│       └── pushing_task_iou_calculator.py  # IOU evaluator for Task 1 (Planar Pushing)
└── pyproject.toml
```

### 3D Models (`models/`)

The following objects will be available in the robot's environment during **Task 1**:

**Note about the models:**
- The cube and cylinder models have a hexagonal extrude used for resetting.
- Evaluation is performed only on the bottom face of the model.
- The models have slots for M3 nuts, to change the weight distribution of the object.
- 3D prints of the models will be in red PLA.

| Model | Filename | Description |
|-------|----------|-------------|
| Square | `square_base_icra_comp.stl` | Square prism with face area of $30mm \times 30mm$ |
| Circle | `circle_base_icra_comp.stl` | Cylinder with base diameter of $30mm$ |
| T-Shape | `t_base_icra_comp.stl` | T-shaped extrude with stroke width of 10mm and overall bounds of $30mm \times 30mm$ |

### Camera Calibration (`config/`)

Per-robot fisheye camera intrinsic parameters are provided for robots. Each YAML file contains the camera matrix `K` and distortion coefficients `D` for the base camera of that robot.

### IOU Evaluator (`src/`)

The Integral of Intersection over Union (IOU) over time during the first $120s$ of the policy execution is used as the score for an evaluation run. The `PushingTaskIOU_Calculator` class (importable as `from rgmc_cloud_robotics_2026 import PushingTaskIOU_Calculator` after installation) can be run on a still image from the base camera to compute the IOU between the detected (red) object and a given target contour.

Key method:

| Method | Description |
|--------|-------------|
| `calculate_iou_from_undistorted_base(img, target_contour, obj_corners)` | Compute IOU from a pre-undistorted base-camera image |

---

## Installation

Install the package in editable mode from the repo root:

```bash
pip install -e .
```

This installs the `rgmc-cloud-robotics-2026` package and its dependencies (`opencv-python`, `numpy`, `pyyaml`) into your environment.

---

## Getting Started

### 1. Instantiate the IOU calculator

```python
from rgmc_cloud_robotics_2026 import PushingTaskIOU_Calculator
from rgmc_cloud_robotics_2026.shapes import SQUARE_CORNERS, CIRCLE_CORNERS, T_CORNERS

# Replace with the YAML for your assigned robot (e.g. cr01)
calculator = PushingTaskIOU_Calculator("config/base-camera-calibration/camera-params-cr01.yaml")
```

### 2. Define the target contour

The target contour is provided directly by the robot API as part of the task specification. 

For local testing or offline evaluation, you can generate an equivalent contour manually using `generate_target_contour`, which projects the object shape to undistorted image coordinates given an offset (in mm) from the workspace centre and an optional rotation angle:

```python
import numpy as np

# Example - in practice, read the target contour from the robot API
target_contour = calculator.generate_target_contour(
    SQUARE_CORNERS,
    offset_from_center_mm=[-40, 30],
    rotation_angle_rad=0.0,
)
```

### 3. Compute IOU from a base-camera image

```python
import cv2

# Load a raw (distorted) robot base camera image
img = cv2.imread("sample_images/base_image_cr01_square.png")

iou, vis_distorted, *_ = calculator.calculate_iou_from_distorted_base(
    img, target_contour, SQUARE_CORNERS
)

print(f"IOU: {iou:.4f}   |   Object angle: {alignment_angle_deg:.1f}°")

# Visualise — green contour = target, white fill = intersection
cv2.imshow("IOU result", vis_distorted)
cv2.waitKey(0)
```

---

## Competition Links

- **CloudGripper Competition Page**: [https://cloudgripper.org/icra2026/index.html](https://cloudgripper.org/icra2026/index.html)
- **RGMC Competition Description**: [https://sites.google.com/view/rgmcomp](https://sites.google.com/view/rgmcomp)
- **CloudGripper Platform**: [https://cloudgripper.org](https://cloudgripper.org)

---

## Contact

For questions about the competition, contact: **info (at) cloudgripper (dot) org**
