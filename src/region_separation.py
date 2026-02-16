import cv2
import numpy as np

# --- Separate abdomen and spine regions ---
def sep_abdomen_spine(image, spider_mask):
    """
    Separate abdomen and spine regions from spider mask

    Args:
        image (np.array): RGB image of spider
        spider_mask (np.array): Binary mask of spider (0=background, 1=spider)

    Returns:
        tuple: (abdomen_mask, final_spine_mask) as binary masks
    """

    # Convert RGB to LAB and HSV
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, _, _ = cv2.split(lab)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Enhance brightness contrast using CLAHE (improve separation between abdomen and spine)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_boosted = clahe.apply(l_channel)
    spider_l = cv2.bitwise_and(l_boosted, l_boosted, mask=spider_mask.astype(np.uint8)) # Only spider pixels in enhanced region kept
    
    ghost_abdomen = np.zeros_like(l_channel)

    # Otsu's thresholding to separate bright abdomen areas from darker structures
    pixels = spider_l[spider_l > 0]
    if len(pixels) > 0:
        thresh_val, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bright_mask = cv2.threshold(l_boosted, thresh_val * 0.9, 255, cv2.THRESH_BINARY)
        
        combined_body = cv2.bitwise_and(bright_mask, bright_mask, mask=spider_mask.astype(np.uint8))

        # Find connected components in bright body regions
        contours, _ = cv2.findContours(combined_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Keep only largest bright region as abdomen
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(ghost_abdomen, [largest_cnt], -1, 255, thickness=cv2.FILLED)
            
            # Morphological closing to smooth abdomen mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            ghost_abdomen = cv2.morphologyEx(ghost_abdomen, cv2.MORPH_CLOSE, kernel)
    
    # Final abdomen mask (binary)
    abdomen_mask = cv2.bitwise_and(ghost_abdomen, ghost_abdomen, mask=(combined_body // 255).astype(np.uint8))
    abdomen_mask = (abdomen_mask > 0).astype(np.uint8)

    # Subtract abddomen from full spider mask
    not_abdomen = cv2.subtract(spider_mask.astype(np.uint8), (ghost_abdomen // 255).astype(np.uint8))
    not_abdomen[not_abdomen < 0] = 0
    
    # Identify red regions in HSV space (spines often red)
    lower_red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 70, 40]), np.array([12, 255, 255])),
                               cv2.inRange(hsv, np.array([160, 70, 40]), np.array([180, 255, 255])))
    
    # Combine non-abdomen and red areas to find spine candidates
    spine_candidates = cv2.bitwise_or(not_abdomen, (lower_red // 255).astype(np.uint8))
    spine_candidates = cv2.bitwise_and(spine_candidates, spider_mask.astype(np.uint8))
    spine_candidates[ghost_abdomen > 0] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spine_candidates)
    final_spine_mask = np.zeros_like(spine_candidates)

    # Calculate approximate vertical center for spatial filtering
    coords = np.where(spider_mask > 0)
    a_cy = int(np.mean(coords[0])) if len(coords[0]) > 0 else 0

    # Filter components based on area and position
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 35: # Slightly lower area to catch thin spine tips
            comp_mask = (labels == i).astype(np.uint8)
            is_red = cv2.bitwise_and(comp_mask, (lower_red // 255).astype(np.uint8)).any()

            # If it's above abdomen center or clearly red, it's a spine
            if centroids[i][1] < (a_cy + 65) or is_red:
                final_spine_mask[labels == i] = 1

    return abdomen_mask, final_spine_mask

def get_segmented_spider_image(image, spider_mask):
    """
    Apply binary mask to an image to isolate spider from background

    Args:
        image (np.array): original RGB image
        spider_mask (np.array): binary mask where spider pixels are non-0
    """
    segmented_image = cv2.bitwise_and(image, image, mask=spider_mask) # Pixels outside the mask are set to black
    return segmented_image