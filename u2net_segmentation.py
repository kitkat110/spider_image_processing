import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model.u2net import U2NETP  # from U2-Net repo

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