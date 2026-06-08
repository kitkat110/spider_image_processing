import logging

import numpy as np

# Reference colors in HSV (hue, saturation, value)
REF_COLORS = {
    "white":  np.array([0,   0,   255]),
    "black":  np.array([0,   0,   0]),
    "red":    np.array([0,   200, 200]),
    "orange": np.array([20,  200, 200]),
    "yellow": np.array([30,  200, 200]),
    "green":  np.array([60,  200, 200]),
    "blue":   np.array([110, 200, 200]),
}


def classify_color_distance(pixel_hsv: np.ndarray) -> str:
    """
    Classify a pixel's color by finding the closest reference color in HSV space using weighted Euclidean distance.
    Saturation and value are weighted more heavily than hue to better distinguish true dark colors from shadowed bright ones

    Args:
        pixel_hsv: HSV values of a single pixel as a NumPy array

    Returns:
        Name of the closest reference color
    """
    logging.debug(f"Classifying color for HSV pixel: {pixel_hsv}")

    weights = np.array([1, 2, 3])
    min_dist = float('inf')
    best_color = "unknown"

    for color_name, ref_hsv in REF_COLORS.items():
        dist = np.linalg.norm((pixel_hsv - ref_hsv) * weights)
        if dist < min_dist:
            min_dist = dist
            best_color = color_name

    logging.debug(f"Classified as: {best_color}")
    return best_color

