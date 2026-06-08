import logging

import cv2
import numpy as np

def sep_abdomen_spine(image: np.ndarray, spider_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate abdomen and spine regions from a spider mask

    Args:
        image: RGB image of spider as a NumPy array
        spider_mask: Binary mask of spider (0=background, 1=spider)

    Returns:
        Tuple of (abdomen_mask, spine_mask) as binary NumPy arrays
    """
    logging.debug("Separating abdomen and spine regions")

    # Convert RGB to LAB and HSV
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, _, _ = cv2.split(lab)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Enhance brightness contrast using CLAHE to improve separation between abdomen and spine
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_boosted = clahe.apply(l_channel)

    # Keep only spider pixels in enhanced brightness channel
    spider_l = cv2.bitwise_and(l_boosted, l_boosted, mask=spider_mask.astype(np.uint8))

    ghost_abdomen = np.zeros_like(l_channel)
    combined_body = np.zeros_like(l_channel)

    # Otsu's thresholding to separate bright abdomen areas from darker structures
    pixels = spider_l[spider_l > 0]
    if len(pixels) > 0:
        thresh_val, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bright_mask = cv2.threshold(l_boosted, thresh_val * 0.9, 255, cv2.THRESH_BINARY)

        combined_body = cv2.bitwise_and(bright_mask, bright_mask, mask=spider_mask.astype(np.uint8))

        # Keep only the largest bright region as the abdomen
        contours, _ = cv2.findContours(combined_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(ghost_abdomen, [largest_cnt], -1, 255, thickness=cv2.FILLED)

            # Morphological closing to smooth abdomen mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            ghost_abdomen = cv2.morphologyEx(ghost_abdomen, cv2.MORPH_CLOSE, kernel)
        else:
            logging.warning("No contours found for abdomen segmentation")

    # Final abdomen mask (binary)
    abdomen_mask = cv2.bitwise_and(ghost_abdomen, ghost_abdomen, mask=(combined_body // 255).astype(np.uint8))
    abdomen_mask = (abdomen_mask > 0).astype(np.uint8)

    # Subtract abdomen from full spider mask to get non-abdomen region
    not_abdomen = cv2.subtract(spider_mask.astype(np.uint8), (ghost_abdomen // 255).astype(np.uint8))
    not_abdomen[not_abdomen < 0] = 0

    # Identify red regions in HSV space (spines are often red)
    lower_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 70, 40]),   np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([160, 70, 40]), np.array([180, 255, 255]))
    )

    # Combine non-abdomen and red regions to find spine candidates
    spine_candidates = cv2.bitwise_or(not_abdomen, (lower_red // 255).astype(np.uint8))
    spine_candidates = cv2.bitwise_and(spine_candidates, spider_mask.astype(np.uint8))
    spine_candidates[ghost_abdomen > 0] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spine_candidates)
    final_spine_mask = np.zeros_like(spine_candidates)

    # Calculate approximate vertical center of spider for spatial filtering
    coords = np.where(spider_mask > 0)
    a_cy = int(np.mean(coords[0])) if len(coords[0]) > 0 else 0

    # Keep components that are large enough and above the abdomen center or clearly red
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 35:
            comp_mask = (labels == i).astype(np.uint8)
            is_red = cv2.bitwise_and(comp_mask, (lower_red // 255).astype(np.uint8)).any()
            if centroids[i][1] < (a_cy + 65) or is_red:
                final_spine_mask[labels == i] = 1

    logging.debug(f"Abdomen pixels: {abdomen_mask.sum()}, spine pixels: {final_spine_mask.sum()}")
    return abdomen_mask, final_spine_mask

def get_segmented_spider_image(image: np.ndarray, spider_mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to isolate the spider from the background

    Args:
        image: Original RGB image as a NumPy array
        spider_mask: Binary mask where spider pixels are non-zero

    Returns:
        RGB image with background set to black outside the mask
    """
    logging.debug("Applying spider mask to image")
    return cv2.bitwise_and(image, image, mask=spider_mask)