import argparse
import logging

from src.preprocessing import load_images
from src.u2net_segmentation import load_u2net_model, segment_spider_u2net
from src.region_separation import sep_abdomen_spine, get_segmented_spider_image
from src.features import extract_abdomen_color, extract_spine_color
from utils.logging_config import setup_logging, add_logging_argument

def process_spider_image(img_path: str, u2net_model: object, device: str="cpu") -> dict:
    """
    Process a spider image through segmentation and feature extraction.

    Args:
        img_path: Path to the spider image.
        u2net_model: Loaded U2Net model.
        device: Device to run model on.

    Returns:
        dict: Dictionary containing extracted color features, segmented image, and masks.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Staring spider image processing: {img_path}")

    # Try to load images 
    try:
        image = load_images(img_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {img_path}")
    except Exception as e:
        logger.exception(f"Unexpected error loading image: {img_path}")
        return None

    # Segmentation
    spider_mask = segment_spider_u2net(image, u2net_model, device=device)
    abdomen_mask, spine_mask = sep_abdomen_spine(image, spider_mask)

    # Feature extraction
    abdomen_color = extract_abdomen_color(image, abdomen_mask)
    spine_color = extract_spine_color(image, spine_mask)

    # Get segmented image
    segmented_image = get_segmented_spider_image(image, spider_mask)

    logger.info(f"Finished processing spider image: {img_path}")

    return {
        "abdomen_color": abdomen_color,
        "spine_color": spine_color,
        "segmented_image": segmented_image,
        "abdomen_mask": abdomen_mask,
        "spine_mask": spine_mask,
        "spider_mask": spider_mask
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_logging_argument(parser)

    parser.add_argument("--image", required=True, help="Path to spider image")
    parser.add_argument("--device", default="cpu", help="Device")

    args = parser.parse_args()

    setup_logging(args.loglevel)
    logger = logging.getLogger(__name__)
    logger.info("Script started")

    # Load U2Net model
    model_path = "saved_models/u2netp/u2netp.pth"
    logger.info(f"Loading U2Net model from {model_path}")
    u2net_model = load_u2net_model(model_path, device=args.device)
    logger.info(f"U2Net model loaded on device: {args.device}")

    # Process spider image
    results = process_spider_image(args.image, u2net_model, args.device)
    if results is None:
        logger.warning(f"Processing failed for image: {args.image}")
    else:
        logger.info(f"Processing complete for image: {args.image}")