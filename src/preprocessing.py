import cv2
import numpy as np

# --- Load and preprocess images ---
def load_images(img_path: str, size=512): 
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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)

    # Merge back the enhanced L-channel with original A and B channels
    img_enhanced = cv2.merge((l_enhanced, a, b))

    return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)