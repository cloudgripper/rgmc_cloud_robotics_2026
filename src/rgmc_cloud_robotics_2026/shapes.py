import numpy as np

# ================================
# Object Dimensions (in mm)
# ================================
SQUARE_WIDTH_MM = 30.0
SQUARE_HEIGHT_MM = 30.0
SQUARE_THICKNESS_MM = 10.0

CIRCLE_RADIUS_MM = 15.0
CIRCLE_NUM_POINTS = 64

T_TOTAL_WIDTH_MM = 30.0
T_TOTAL_HEIGHT_MM = 30.0
T_THICKNESS_MM = 10.0

# ================================
# Square Corners
# ================================
SQUARE_CORNERS = np.array([
    [-SQUARE_WIDTH_MM/2, -SQUARE_HEIGHT_MM/2, 0],  # Top-left
    [SQUARE_WIDTH_MM/2, -SQUARE_HEIGHT_MM/2, 0],   # Top-right
    [SQUARE_WIDTH_MM/2, SQUARE_HEIGHT_MM/2, 0],    # Bottom-right
    [-SQUARE_WIDTH_MM/2, SQUARE_HEIGHT_MM/2, 0],   # Bottom-left
], dtype=np.float32)

# ================================
# Circle Corners
# ================================
_angles = np.linspace(0, 2 * np.pi, CIRCLE_NUM_POINTS, endpoint=False)
CIRCLE_CORNERS = np.array([
    [CIRCLE_RADIUS_MM * np.cos(angle), CIRCLE_RADIUS_MM * np.sin(angle), 0]
    for angle in _angles
], dtype=np.float32)

# ================================
# T Corners
# ================================
T_CORNERS = np.array([
    # Right Vertical Bar
    [ T_TOTAL_WIDTH_MM/2 - T_THICKNESS_MM,         -T_TOTAL_HEIGHT_MM/2, 0],  # P1: Top-Left of Bar
    [ T_TOTAL_WIDTH_MM/2,                          -T_TOTAL_HEIGHT_MM/2, 0],  # P2: Top-Right Outer
    [ T_TOTAL_WIDTH_MM/2,                           T_TOTAL_HEIGHT_MM/2, 0],  # P3: Bottom-Right Outer
    [ T_TOTAL_WIDTH_MM/2 - T_THICKNESS_MM,          T_TOTAL_HEIGHT_MM/2, 0],  # P4: Bottom-Left of Bar

    # Horizontal Stem (Connection)
    [ T_TOTAL_WIDTH_MM/2 - T_THICKNESS_MM,          T_THICKNESS_MM/2,    0],  # P5: Stem Bottom-Right
    [-T_TOTAL_WIDTH_MM/2,                           T_THICKNESS_MM/2,    0],  # P6: Stem Bottom-Left Outer
    [-T_TOTAL_WIDTH_MM/2,                          -T_THICKNESS_MM/2,    0],  # P7: Stem Top-Left Outer
    [ T_TOTAL_WIDTH_MM/2 - T_THICKNESS_MM,         -T_THICKNESS_MM/2,    0],  # P8: Stem Top-Right
], dtype=np.float32)
