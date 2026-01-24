# Libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.u2net import U2NETP  # from U2-Net repo

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


# --- Load and preprocess images ---
def load_images(img_path, size=512):
    """
    Ensure all images are same size and enhanced

    Args:
        img_path (str): Path to the image file
        size (int): Desired width and height to resize the image

    Returns:
        np.array: Preprocessed RGB image as a NumPy array
    """

    img = cv2.imread(img_path)
    if img is None:
        return None

    # Convert from OpenCV's default BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))

    # Convert RGB to LAB (separate lightness from color channels)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)

    # Merge back the enhanced L-channel with original A and B channels
    img_enhanced = cv2.merge((l_enhanced, a, b))

    return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)


# --- U2-Net model integration ---
def load_u2net_model(model_path, device="cpu"):
    """
    Load pre-trained U2NETP model (machine learning model for background removal)

    Args:
        model_path (str): Path to saved model weights
        device (str): Device to load the model onto

    Returns:
        torch.nn.Module: Loaded U2NETP model
    """

    model = U2NETP(3, 1) # 3 channels (RGB), 1 output channel (mask)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_u2net_mask(model, image, device="cpu", size=320):
    """
    Generate foreground mask using U2NETP model for given image

    Args:
        model (torch.nn.Module): Pre-trained U2NETP model
        image (np.array): RGB image as NumPy array
        device (str): Device to run the model on
        size (int): Size to resize image for model input

    Returns:
        np.array: Binary segmentation mask (0=background, 1=foreground)
    """

    orig_h, orig_w = image.shape[:2]

    # Convert NumPy array to PIL image for torchvision transforms
    pil = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    inp = transform(pil).unsqueeze(0).to(device)
    
    # Run model inference (no gradients)
    with torch.no_grad():
        d1, *_ = model(inp)

    # Convert output tensor to NumPy array
    mask = d1[0,0].cpu().numpy()
    
    # Normalize mask to [0, 1] and resize to original image size
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = cv2.resize(mask, (orig_w, orig_h))

    mask_binary = (mask > 0.5).astype(np.uint8)

    return mask_binary


# --- Spider segmentation using U2-Net ---
def segment_spider_u2net(image, model, device="cpu"):
    """
    Segment spider from background using U2NETP and post-processing

    Args:
        image (np.array): RGB image of spider
        model (torch.nn.Module): Pre-trained U2NETP model
        device (str): Device to run the model on

    Returns:
        np.array: Cleaned binary mask of spider (0=background, 1=spider)
    """

    # Initial foreground mask prediction
    fg_mask = predict_u2net_mask(model, image, device)
    
    # Morphological closing to fill small holes in mask
    kernel = np.ones((5,5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours in mask
    # Keep only largest connected component (assumed to be spider)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(fg_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        # Fill in largest contour on clean mask
        cv2.drawContours(clean_mask, [largest], -1, 1, thickness=cv2.FILLED)
    
    return clean_mask


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


# --- Color classification ---
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

def get_segmented_spider_image(image, spider_mask):
    """
    Apply binary mask to an image to isolate spider from background

    Args:
        image (np.array): original RGB image
        spider_mask (np.array): binary mask where spider pixels are non-0
    """
    segmented_image = cv2.bitwise_and(image, image, mask=spider_mask) # Pixels outside the mask are set to black
    return segmented_image


# --- Main processing function ---
def process_spider_image(img_path, u2net_model, device="cpu"):
    image = load_images(img_path)
    spider_mask = segment_spider_u2net(image, u2net_model, device=device)
    abdomen_mask, spine_mask = sep_abdomen_spine(image, spider_mask)
    abdomen_color = extract_abdomen_color(image, abdomen_mask)
    spine_color = extract_spine_color(image, spine_mask)
    segmented_image = get_segmented_spider_image(image, spider_mask)

    return {
        "abdomen_color": abdomen_color,
        "spine_color": spine_color,
        "segmented_image": segmented_image,
        "abdomen_mask": abdomen_mask,
        "spine_mask": spine_mask,
        "spider_mask": spider_mask
    }

# --- Test code ---
if __name__=="__main__":
    device = "cpu"
    model_path = os.path.join(BASE_DIR, "saved_models", "u2netp", "u2netp.pth")
    u2net_model = load_u2net_model(model_path, device)

    img_path = "/Users/minimal_kitkat/Desktop/spiny_orb4.jpg"
    results = process_spider_image(img_path, u2net_model, device=device)

    print("Abdomen Color:", results["abdomen_color"])
    print("Spine Color:", results["spine_color"])

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title("Segmented Spider")
    plt.imshow(results["segmented_image"])
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Abdomen Mask")
    plt.imshow(results["abdomen_mask"], cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Spine Mask")
    plt.imshow(results["spine_mask"], cmap='gray')
    plt.axis('off')
    plt.show()
