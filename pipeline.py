"""
Runs the full spider processing pipeline over raw images fetched by get_observations.py: preprocessing → segmentation → color extraction. 
All results are merged back into the original observations DataFrame and saved as a single CSV

Usage:
    python pipeline.py --input_csv data/observations.csv
    python pipeline.py -l DEBUG --device cuda
"""

import argparse
import logging
import socket
from pathlib import Path

import cv2
import pandas as pd

from src.preprocessing import load_images
from src.u2net_segmentation import load_u2net_model, segment_spider_u2net
from src.region_separation import sep_abdomen_spine, get_segmented_spider_image
from src.features import extract_abdomen_color, extract_spine_color

# -------------------------
# Logging setup
# -------------------------
parser = argparse.ArgumentParser(
    description="Run full spider pipeline over raw observations from get_observations.py."
)
parser.add_argument('-l', '--loglevel',
                    type=str,
                    required=False,
                    default='WARNING',
                    help='set log level to DEBUG, INFO, WARNING, ERROR, or CRITICAL')
parser.add_argument('--input_csv', default='data/observations.csv',
                    help='CSV produced by get_observations.py (default: data/observations.csv)')
parser.add_argument('--device', default='cpu',
                    help="PyTorch device: 'cpu' or 'cuda' (default: cpu)")
parser.add_argument('--model_path', default='saved_models/u2netp/u2netp.pth',
                    help='Path to U2Net weights file')
parser.add_argument('--segmented_dir', default='data/segmented',
                    help='Directory to save segmented images (default: data/segmented)')
parser.add_argument('--output_csv', default='data/results.csv',
                    help='Where to save results (default: data/results.csv)')
args = parser.parse_args()

format_str = (
    f'[%(asctime)s {socket.gethostname()}] '
    '%(filename)s:%(funcName)s:%(lineno)s - %(levelname)s: %(message)s'
)
logging.basicConfig(level=args.loglevel, format=format_str)

# -------------------------
# Functions
# -------------------------

def process_spider_image(img_path: str, u2net_model: object, device: str = "cpu") -> dict | None:
    """
    Preprocess a raw image and run it through the full spider pipeline.

    Preprocessing (resize + CLAHE contrast enhancement) is applied here via
    load_images() before segmentation and feature extraction.

    Args:
        img_path:    Path to the raw image saved by get_observations.py.
        u2net_model: Loaded U2Net model.
        device:      Device to run model on ('cpu' or 'cuda').

    Returns:
        Dict with keys: abdomen_color, spine_color, segmented_image,
        abdomen_mask, spine_mask, spider_mask. Returns None on failure.
    """
    logging.debug(f"Processing image: {img_path}")

    image = load_images(img_path)
    if image is None:
        logging.error(f"Could not load image: {img_path}")
        return None

    spider_mask              = segment_spider_u2net(image, u2net_model, device=device)
    abdomen_mask, spine_mask = sep_abdomen_spine(image, spider_mask)
    segmented_image          = get_segmented_spider_image(image, spider_mask)

    return {
        "abdomen_color":   extract_abdomen_color(image, abdomen_mask),
        "spine_color":     extract_spine_color(image, spine_mask),
        "segmented_image": segmented_image,
        "abdomen_mask":    abdomen_mask,
        "spine_mask":      spine_mask,
        "spider_mask":     spider_mask,
    }

def run_pipeline(df: pd.DataFrame, model_path: str, device: str, segmented_dir: Path) -> pd.DataFrame:
    """
    Run preprocessing, segmentation, and feature extraction on all successfully
    downloaded images. Results are appended as new columns to the input DataFrame.

    Args:
        df:            DataFrame from get_observations.py (download_status == 'ok').
        model_path:    Path to the U2Net .pth weights file.
        device:        PyTorch device string ('cpu' or 'cuda').
        segmented_dir: Directory to save segmented output images.

    Returns:
        DataFrame with all original metadata columns plus:
            abdomen_color, spine_color, segmented_image_path,
            abdomen_mask, spine_mask, spider_mask, processing_status
    """
    logging.info(f"Loading U2Net model from {model_path}")
    u2net_model = load_u2net_model(model_path, device=device)
    logging.info(f"Model loaded on device: {device}")

    segmented_dir.mkdir(parents=True, exist_ok=True)

    abdomen_colors  = []
    spine_colors    = []
    segmented_paths = []
    abdomen_masks   = []
    spine_masks     = []
    spider_masks    = []
    statuses        = []

    for i, (idx, row) in enumerate(df.iterrows()):
        logging.debug(f"Processing image {i+1}/{len(df)} — obs {row['observation_id']}")
        results = process_spider_image(row["local_path"], u2net_model, device)

        if results is None:
            logging.warning(f"Processing failed for obs {row['observation_id']}")
            abdomen_colors.append(None)
            spine_colors.append(None)
            segmented_paths.append(None)
            abdomen_masks.append(None)
            spine_masks.append(None)
            spider_masks.append(None)
            statuses.append("failed")
        else:
            src_stem = Path(row["local_path"]).stem
            seg_path = segmented_dir / f"{src_stem}_segmented.png"
            cv2.imwrite(str(seg_path), cv2.cvtColor(results["segmented_image"], cv2.COLOR_RGB2BGR))
            logging.debug(f"Saved segmented image: {seg_path}")

            abdomen_colors.append(results["abdomen_color"])
            spine_colors.append(results["spine_color"])
            segmented_paths.append(str(seg_path))
            abdomen_masks.append(results["abdomen_mask"])
            spine_masks.append(results["spine_mask"])
            spider_masks.append(results["spider_mask"])
            statuses.append("ok")

    df = df.copy()
    df["abdomen_color"]        = abdomen_colors
    df["spine_color"]          = spine_colors
    df["segmented_image_path"] = segmented_paths
    df["abdomen_mask"]         = abdomen_masks
    df["spine_mask"]           = spine_masks
    df["spider_mask"]          = spider_masks
    df["processing_status"]    = statuses

    logging.info(f"Pipeline complete: {statuses.count('ok')}/{len(df)} images processed successfully")
    return df

def main():
    logging.info("Starting spider image processing pipeline")

    df = pd.read_csv(args.input_csv)
    logging.info(f"Loaded {len(df)} rows from {args.input_csv}")

    ready = df[df["download_status"] == "ok"].copy()
    skipped = len(df) - len(ready)
    if skipped:
        logging.warning(f"Skipping {skipped} rows with download_status != 'ok'")

    if ready.empty:
        logging.critical("No successfully downloaded images to process. Exiting.")
        return

    results_df = run_pipeline(
        df=ready,
        model_path=args.model_path,
        device=args.device,
        segmented_dir=Path(args.segmented_dir),
    )

    # Masks are numpy arrays — drop before saving to CSV
    csv_df = results_df.drop(columns=["abdomen_mask", "spine_mask", "spider_mask"])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved results to {args.output_csv}")

    print(f"\nSaved {len(csv_df)} rows to {args.output_csv}")
    print(csv_df[[
        "observation_id", "latitude", "longitude", "observed_on",
        "abdomen_color", "spine_color",
        "segmented_image_path", "processing_status"
    ]].to_string())

    logging.info("Spider image processing pipeline complete")

if __name__ == '__main__':
    main()