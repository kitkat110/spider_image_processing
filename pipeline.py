"""
Runs the full spider processing pipeline over raw images fetched by
get_observations.py: preprocessing → segmentation → color extraction.
All results are merged back into the original observations DataFrame
and saved as a single CSV.
 
Usage:
    python inaturalist_pipeline.py --input_csv data/observations.csv --device cpu
"""
 
import argparse
import logging
from pathlib import Path
 
import cv2
import pandas as pd
 
from src.preprocessing import load_images
from src.u2net_segmentation import load_u2net_model, segment_spider_u2net
from src.region_separation import sep_abdomen_spine, get_segmented_spider_image
from src.features import extract_abdomen_color, extract_spine_color
from utils.logging_config import setup_logging, add_logging_argument
 

# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
 
def process_spider_image(img_path: str, u2net_model: object, device: str = "cpu") -> dict | None:
    """
    Preprocess a raw image and run it through the full spider pipeline.
 
    Preprocessing (resize + CLAHE contrast enhancement) is applied here
    via load_images() before segmentation and feature extraction.
 
    Args:
        img_path:    Path to the raw image saved by get_observations.py.
        u2net_model: Loaded U2Net model.
        device:      Device to run model on ('cpu' or 'cuda').
 
    Returns:
        Dict with keys: abdomen_color, spine_color, segmented_image,
        abdomen_mask, spine_mask, spider_mask. Returns None on failure.
    """
    logger = logging.getLogger(__name__)
 
    # Preprocessing: resize + CLAHE
    image = load_images(img_path)
    if image is None:
        logger.warning(f"Could not load image: {img_path}")
        return None
 
    # Segmentation
    spider_mask              = segment_spider_u2net(image, u2net_model, device=device)
    abdomen_mask, spine_mask = sep_abdomen_spine(image, spider_mask)
    segmented_image          = get_segmented_spider_image(image, spider_mask)
 
    # Feature extraction
    return {
        "abdomen_color":   extract_abdomen_color(image, abdomen_mask),
        "spine_color":     extract_spine_color(image, spine_mask),
        "segmented_image": segmented_image,
        "abdomen_mask":    abdomen_mask,
        "spine_mask":      spine_mask,
        "spider_mask":     spider_mask,
    }
 
 
# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
 
def run_pipeline(
    input_csv: str,
    device: str,
    model_path: str,
    segmented_dir: str = "data/segmented",
) -> pd.DataFrame:
    """
    Load the observations CSV from get_observations.py, run each raw image
    through preprocessing → segmentation → feature extraction, and return
    the full DataFrame with all results appended.
 
    Segmented images are saved to `segmented_dir` and their paths stored
    in `segmented_image_path`. Every row retains its original location,
    date, and photo metadata alongside the extracted features.
 
    Args:
        input_csv:     Path to the CSV produced by get_observations.py.
        device:        PyTorch device string ('cpu' or 'cuda').
        model_path:    Path to the U2Net .pth weights file.
        segmented_dir: Directory to save segmented output images.
 
    Returns:
        DataFrame with all original metadata columns plus:
            abdomen_color, spine_color,
            segmented_image_path,
            abdomen_mask, spine_mask, spider_mask,
            processing_status
    """
    logger = logging.getLogger(__name__)
 
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} rows from {input_csv}")
 
    ready = df[df["download_status"] == "ok"].copy()
    skipped = len(df) - len(ready)
    if skipped:
        logger.warning(f"Skipping {skipped} rows with download_status != 'ok'")
 
    seg_dir = Path(segmented_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
 
    logger.info(f"Loading U2Net model from {model_path}")
    u2net_model = load_u2net_model(model_path, device=device)
    logger.info(f"Model ready on device: {device}")
 
    abdomen_colors  = []
    spine_colors    = []
    segmented_paths = []
    abdomen_masks   = []
    spine_masks     = []
    spider_masks    = []
    statuses        = []
 
    for i, (idx, row) in enumerate(ready.iterrows()):
        logger.info(f"Processing [{i+1}/{len(ready)}] — obs {row['observation_id']}")
        results = process_spider_image(row["local_path"], u2net_model, device)
 
        if results is None:
            abdomen_colors.append(None)
            spine_colors.append(None)
            segmented_paths.append(None)
            abdomen_masks.append(None)
            spine_masks.append(None)
            spider_masks.append(None)
            statuses.append("failed")
        else:
            src_stem = Path(row["local_path"]).stem
            seg_path = seg_dir / f"{src_stem}_segmented.png"
            cv2.imwrite(
                str(seg_path),
                cv2.cvtColor(results["segmented_image"], cv2.COLOR_RGB2BGR)
            )
 
            abdomen_colors.append(results["abdomen_color"])
            spine_colors.append(results["spine_color"])
            segmented_paths.append(str(seg_path))
            abdomen_masks.append(results["abdomen_mask"])
            spine_masks.append(results["spine_mask"])
            spider_masks.append(results["spider_mask"])
            statuses.append("ok")
 
    ready["abdomen_color"]        = abdomen_colors
    ready["spine_color"]          = spine_colors
    ready["segmented_image_path"] = segmented_paths
    ready["abdomen_mask"]         = abdomen_masks
    ready["spine_mask"]           = spine_masks
    ready["spider_mask"]          = spider_masks
    ready["processing_status"]    = statuses
 
    logger.info(f"Done. {statuses.count('ok')}/{len(ready)} images processed successfully.")
    return ready
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full spider pipeline over raw observations from get_observations.py."
    )
    add_logging_argument(parser)
    parser.add_argument(
        "--input_csv", default="data/observations.csv",
        help="CSV produced by get_observations.py (default: data/observations.csv)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: cpu)"
    )
    parser.add_argument(
        "--model_path", default="saved_models/u2netp/u2netp.pth",
        help="Path to U2Net weights file"
    )
    parser.add_argument(
        "--segmented_dir", default="data/segmented",
        help="Directory to save segmented images (default: data/segmented)"
    )
    parser.add_argument(
        "--output_csv", default="data/results.csv",
        help="Where to save results (default: data/results.csv)"
    )
 
    args = parser.parse_args()
    setup_logging(args.loglevel)
 
    results_df = run_pipeline(
        input_csv=args.input_csv,
        device=args.device,
        model_path=args.model_path,
        segmented_dir=args.segmented_dir,
    )
 
    if not results_df.empty:
        # Masks are numpy arrays — drop before saving to CSV
        csv_df = results_df.drop(columns=["abdomen_mask", "spine_mask", "spider_mask"])
        csv_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved {len(csv_df)} rows to {args.output_csv}")
        print(csv_df[[
            "observation_id", "latitude", "longitude", "observed_on",
            "abdomen_color", "spine_color",
            "segmented_image_path", "processing_status"
        ]].to_string())