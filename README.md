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

### 3D Models (`models/`)

The following objects will be available in the robot's environment during **Task 1**:

**Note about the models:**
- The cube and cylinder models have a hexagonal extrude used for resetting. E
- Evaluation is performed only on the bottom face of the model. 
- The models have slots for M3 nuts, to change the weight distribution of the object. 
- 3D prints of the models will be in red PLA. 

| Model | Filename | Description |
|-------|----------|-------------|
| Cube | `square_profile.stl` | Square prism with face area of $30mm \times 30mm$ |
| Cylinder | `circle_profile.stl` | Cylinder with base diameter of $30mm$ |
| T-Shape | `T.stl` | T-shaped extrude with stroke width of 10mm and ovalrall bounds of $30mm \times 30mm$|

### IOU Evaluator

Integral of Intersection over Union (IOU) over time during the first $120s$ of the policy execution is used as a score of the evaluation run. The Intersection over Union (IOU) evaluator can be run on a still image from the base camera to assess the IOU metric between the detected object and the target for Task 1. 

---

## Getting Started

Running the IOU evaluator:

``` bash
[TODO]
```


---

## Competition Links

- **CloudGripper Competition Page**: [https://cloudgripper.org/icra2026/index.html](https://cloudgripper.org/icra2026/index.html)
- **RGMC Competition Description**: [https://sites.google.com/view/rgmcomp](https://sites.google.com/view/rgmcomp)
- **CloudGripper Platform**: [https://cloudgripper.org](https://cloudgripper.org)

---

## Contact

For questions about the competition, contact: **info (at) cloudgripper (dot) org**
