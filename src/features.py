import cv2
import numpy as np

from src.color_classification import classify_color_distance

def extract_abdomen_color(image, mask):
    """
    Determine dominant abdomen color using cleaned mask and HSV analysis

    Args:
        image (np.array): RGB image of spider
        mask (np.array): Binary mask of abdomen region

    Returns:
        str: Classified abdomen color
    """

    # Erode mask slightly to avoid edge artifacts that may contain background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    safe_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    
    if np.sum(safe_mask) == 0:
        safe_mask = mask.astype(np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Select only pixels inside abdomen mask
    pixels = hsv[safe_mask > 0]
    
    # If no pixels extracted, default to black
    if len(pixels) == 0:
        return "black"

    # Normalize brightness
    shell_max_v = np.percentile(pixels[:, 2], 98)
    if shell_max_v < 180 and shell_max_v > 0:
        scale = 230.0 / shell_max_v
        pixels[:, 2] = np.clip(pixels[:, 2] * scale, 0, 255)

    # If substanial fraction of pixels are white, classify as white
    white_pixels = pixels[(pixels[:, 1] < 55) & (pixels[:, 2] > 150)]
    if len(white_pixels) / len(pixels) > 0.35:
        return "white"

    vibrant_pixels = pixels[pixels[:, 1] > 50]
    
    # Use median HSV color for robustness to noise
    if len(vibrant_pixels) > 10: # Ensure we have enough data
        avg_color = np.median(vibrant_pixels, axis=0)
    else:
        avg_color = np.median(pixels, axis=0)

    return classify_color_distance(avg_color.astype(np.uint8))

def extract_spine_color(image, mask):
    """
    Determine spine color by detecting presence of red pixels within spine mask region

    Args:
        image (np.array): RGB image of spider
        mask (np.array): Binary mask of spine region

    Returns:
        str: Classified spine color ("red" or "black")
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    pixels = hsv[mask > 0]
    
    # If no pixels extracted, default to black
    if len(pixels) == 0:
        return "black"

    # Normalize brightness and saturation
    brightest_samples = np.percentile(pixels[:, 2], 98) 
    if brightest_samples < 160 and brightest_samples > 0:
        scale = 210.0 / brightest_samples
        pixels[:, 2] = np.clip(pixels[:, 2] * scale, 0, 255)
        pixels[:, 1] = np.clip(pixels[:, 1] * 1.5, 0, 255) 
    
    # Identify red pixels
    red_pixels = pixels[
        ((pixels[:, 0] < 18) | (pixels[:, 0] > 162)) & # red hue range
        (pixels[:, 1] > 50) & 
        (pixels[:, 2] > 30)
    ]
    
    # Calculate fraction of spine that is red
    red_ratio = len(red_pixels) / len(pixels)
    
    # If more than 20% of mask is red, classify whole thing as red
    if red_ratio > 0.20:
        return "red"
    else:
        return "black"