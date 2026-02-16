import numpy as np

# Reference colors in HSV (Color, intensity, brightness)
REF_COLORS = {
    "white": np.array([0, 0, 255]),
    "black": np.array([0, 0, 0]),
    "red": np.array([0, 200, 200]),
    "orange": np.array([20, 200, 200]),
    "yellow": np.array([30, 200, 200]),
    "green": np.array([60, 200, 200]),
    "blue": np.array([110, 200, 200]),
}

def classify_color_distance(pixel_hsv):
    """
    Classify pixel color by finding closest reference color in HSV space using weighted Euclidean distance

    Args:
        pixel_hsv (np.array): HSV values of pixel

    Returns:
        str: Name of closest reference color
    """

    h, s, v = pixel_hsv
    min_dist = float('inf')
    best_color = "unknown"

    # Iterate through predefined reference colors
    for color_name, ref_hsv in REF_COLORS.items():
        # More weight to saturation and value differences, hue noise is already reduced
        # Helps to distinguish between true dark colors vs. shadowed bright colors
        weights = np.array([1, 2, 3])

        dist = np.linalg.norm((pixel_hsv - ref_hsv) * weights)
        if dist < min_dist:
            min_dist = dist
            best_color = color_name

    return best_color

