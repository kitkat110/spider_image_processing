import logging

import cv2
import numpy as np

from src.color_classification import classify_color_distance


def extract_abdomen_color(image: np.ndarray, mask: np.ndarray) -> str:
    """
    Determine the dominant abdomen color using the abdomen mask and HSV analysis

    Args:
        image: RGB image of spider as a NumPy array
        mask: Binary mask of abdomen region

    Returns:
        Classified abdomen color as a string
    """
    logging.debug("Extracting abdomen color")

    # Erode mask slightly to avoid edge artifacts that may contain background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    safe_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

    if np.sum(safe_mask) == 0:
        safe_mask = mask.astype(np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Select only pixels inside abdomen mask
    pixels = hsv[safe_mask > 0]

    if len(pixels) == 0:
        logging.warning("No abdomen pixels found — defaulting to black")
        return "black"

    # Normalize brightness
    shell_max_v = np.percentile(pixels[:, 2], 98)
    if 0 < shell_max_v < 180:
        scale = 230.0 / shell_max_v
        pixels[:, 2] = np.clip(pixels[:, 2] * scale, 0, 255)

    # If a substantial fraction of pixels are white, classify as white
    white_pixels = pixels[(pixels[:, 1] < 55) & (pixels[:, 2] > 150)]
    if len(white_pixels) / len(pixels) > 0.35:
        logging.debug("Abdomen classified as white")
        return "white"

    # Use median HSV of vibrant pixels for robustness to noise
    vibrant_pixels = pixels[pixels[:, 1] > 50]
    avg_color = np.median(vibrant_pixels, axis=0) if len(vibrant_pixels) > 10 else np.median(pixels, axis=0)

    result = classify_color_distance(avg_color.astype(np.uint8))
    logging.debug(f"Abdomen color: {result}")
    return result


def extract_spine_color(image: np.ndarray, mask: np.ndarray) -> str:
    """
    Determine spine color by detecting the presence of red pixels within the spine mask region

    Args:
        image: RGB image of spider as a NumPy array
        mask: Binary mask of spine region

    Returns:
        Classified spine color as a string ('red' or 'black')
    """
    logging.debug("Extracting spine color")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    pixels = hsv[mask > 0]

    if len(pixels) == 0:
        logging.warning("No spine pixels found — defaulting to black")
        return "black"

    # Normalize brightness and saturation
    brightest_samples = np.percentile(pixels[:, 2], 98)
    if 0 < brightest_samples < 160:
        scale = 210.0 / brightest_samples
        pixels[:, 2] = np.clip(pixels[:, 2] * scale, 0, 255)
        pixels[:, 1] = np.clip(pixels[:, 1] * 1.5, 0, 255)

    # Identify red pixels (hue wraps around 0/180 in OpenCV HSV)
    red_pixels = pixels[
        ((pixels[:, 0] < 18) | (pixels[:, 0] > 162)) &
        (pixels[:, 1] > 50) &
        (pixels[:, 2] > 30)
    ]

    # If more than 20% of spine mask is red, classify as red
    red_ratio = len(red_pixels) / len(pixels)
    result = "red" if red_ratio > 0.20 else "black"

    logging.debug(f"Spine color: {result} (red ratio: {red_ratio:.2f})")
    return result