import logging

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model.u2net import U2NETP  # from U2-Net repo

def load_u2net_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load pre-trained U2NETP model (machine learning model for background removal)

    Args:
        model_path: Path to saved model weights
        device: Device to load the model onto

    Returns:
        Loaded U2NETP model in eval mode
    """
    logging.info(f"Loading U2NETP model from {model_path} on device: {device}")

    model = U2NETP(3, 1)  # 3 input channels (RGB), 1 output channel (mask)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    logging.info("U2NETP model loaded successfully")
    return model

def predict_u2net_mask(model: torch.nn.Module, image: np.ndarray, device: str = "cpu", size: int = 320) -> np.ndarray:
    """
    Generate a foreground mask using the U2NETP model for a given image

    Args:
        model: Pre-trained U2NETP model
        image: RGB image as a NumPy array
        device: Device to run the model on
        size: Size to resize image for model input

    Returns:
        Binary segmentation mask as a NumPy array (0=background, 1=foreground)
    """
    logging.debug("Predicting foreground mask with U2NETP")

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
    mask = d1[0, 0].cpu().numpy()

    # Normalize mask to [0, 1] and resize to original image dimensions
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = cv2.resize(mask, (orig_w, orig_h))

    return (mask > 0.5).astype(np.uint8)

def segment_spider_u2net(image: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    """
    Segment spider from background using U2NETP and post-processing

    Args:
        image: RGB image of spider as a NumPy array
        model: Pre-trained U2NETP model
        device: Device to run the model on

    Returns:
        Cleaned binary mask of spider as a NumPy array (0=background, 1=spider)
    """
    logging.debug("Segmenting spider with U2NETP")

    # Initial foreground mask prediction
    fg_mask = predict_u2net_mask(model, image, device)

    # Morphological closing to fill small holes in mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component (assumed to be the spider)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(fg_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 1, thickness=cv2.FILLED)
    else:
        logging.warning("No contours found in segmentation mask — returning empty mask")

    return clean_mask