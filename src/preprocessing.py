import logging

import cv2
import numpy as np

def load_images(img_path: str, size: int = 512) -> np.ndarray | None:
    """
    Load a raw image from disk, resize it, and enhance contrast using CLAHE

    Args:
        img_path: Path to the image file
        size: Desired width and height in pixels to resize the image

    Returns:
        Preprocessed RGB image as a NumPy array, or None if the image could not be loaded
    """
    logging.debug(f"Loading image: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"Could not read image: {img_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back the enhanced L-channel with original A and B channels
    img_enhanced = cv2.merge((l_enhanced, a, b))

    logging.debug(f"Finished preprocessing image: {img_path}")
    return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)