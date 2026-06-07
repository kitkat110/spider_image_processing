"""
Retrieves Gasteracantha cancriformis (Spinyback Orbweaver) observations from iNaturalist and saves raw images + metadata to disk

Usage:
    python get_observations.py --count 20 --output_dir data/observations
    python get_observations.py -l DEBUG --count 50
"""

import argparse
import logging
import socket
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

# -------------------------
# CONSTANTS
# -------------------------
INAT_API = "https://api.inaturalist.org/v1/observations"
TAXON_ID = 49540  # Gasteracantha cancriformis on iNaturalist

# -------------------------
# Logging setup
# -------------------------
parser = argparse.ArgumentParser(
    description="Fetch and download raw Gasteracantha cancriformis observations from iNaturalist"
)
parser.add_argument('-l', '--loglevel',
                    type=str,
                    required=False,
                    default='WARNING',
                    help='set log level to DEBUG, INFO, WARNING, ERROR, or CRITICAL')
parser.add_argument('--count', type=int, default=20,
                    help='Number of photos to fetch (default: 20)')
parser.add_argument('--output_dir', default='data/observations',
                    help='Directory to save raw images (default: data/observations)')
parser.add_argument('--output_csv', default='data/observations.csv',
                    help='Where to save the metadata CSV (default: data/observations.csv)')
args = parser.parse_args()

format_str = (
    f'[%(asctime)s {socket.gethostname()}] '
    '%(filename)s:%(funcName)s:%(lineno)s - %(levelname)s: %(message)s'
)
logging.basicConfig(level=args.loglevel, format=format_str)

# -------------------------
# Functions
# -------------------------
def fetch_inaturalist_observations(count: int = 20) -> pd.DataFrame:
    """
    Fetch up to `count` research-grade observations of Gasteracantha cancriformis from the iNaturalist API

    Args:
        count: Number of photos to retrieve (max 200 per request)

    Returns:
        DataFrame with columns:
            observation_id, photo_id, photo_url, photo_url_large,
            latitude, longitude, place_guess, observed_on,
            quality_grade, user_login, taxon_name, common_name
    """
    logging.info(f"Fetching {count} iNaturalist observations for taxon_id={TAXON_ID}")

    params = {
        "taxon_id": TAXON_ID,
        "quality_grade": "research",
        "photos": "true",
        "per_page": min(count, 200),
        "page": 1,
        "order": "desc",
        "order_by": "created_at",
    }

    response = requests.get(INAT_API, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    rows = []
    for obs in data.get("results", []):
        location = obs.get("location", "")
        lat, lon = (location.split(",") + [None, None])[:2] if location else (None, None)

        photos = obs.get("photos", [])
        if not photos:
            logging.debug(f"No photos for observation {obs.get('id')}, skipping")
            continue
 
        # Take the first photo
        photo = photos[0]
        base_url = photo.get("url", "")
        rows.append({
            "observation_id":  obs.get("id"),
            "photo_id":        photo.get("id"),
            "photo_url":       base_url.replace("square", "medium"),
            "photo_url_large": base_url.replace("square", "large"),
            "latitude":        float(lat) if lat else None,
            "longitude":       float(lon) if lon else None,
            "place_guess":     obs.get("place_guess"),
            "observed_on":     obs.get("observed_on"),
            "quality_grade":   obs.get("quality_grade"),
            "user_login":      obs.get("user", {}).get("login"),
            "taxon_name":      obs.get("taxon", {}).get("name"),
            "common_name":     obs.get("taxon", {}).get("preferred_common_name"),
        })

    logging.info(f"Retrieved {len(rows)} photos from {len(data.get('results', []))} observations")
    return pd.DataFrame(rows)

def download_raw_images(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Download the raw image for each row in the DataFrame and save as PNG. Adds `local_path` and `download_status` columns to the DataFrame

    Args:
        df: DataFrame from fetch_inaturalist_observations()
        output_dir: Directory to save raw images

    Returns:
        Updated DataFrame with `local_path` and `download_status` columns
    """
    logging.info(f"Downloading images to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []
    statuses = []

    for i, row in df.iterrows():
        filename = f"obs{row['observation_id']}_photo{row['photo_id']}.png"
        save_path = output_dir / filename

        if save_path.exists():
            logging.debug(f"Already exists, skipping: {filename}")
            local_paths.append(str(save_path))
            statuses.append("ok")
            continue

        logging.debug(f"Downloading [{i+1}/{len(df)}]: {filename}")
        try:
            resp = requests.get(row["photo_url"], timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(str(save_path))
            local_paths.append(str(save_path))
            statuses.append("ok")
        except Exception as e:
            logging.error(f"Failed to download {row['photo_url']}: {e}")
            local_paths.append(None)
            statuses.append("failed")

        time.sleep(0.5)

    df = df.copy()
    df["local_path"]      = local_paths
    df["download_status"] = statuses

    logging.info(f"Downloaded {statuses.count('ok')}/{len(df)} images successfully")
    return df


def main():
    logging.info("Starting iNaturalist observation fetch")

    df = fetch_inaturalist_observations(count=args.count)

    if df.empty:
        logging.critical("No observations returned from iNaturalist. Exiting.")
        return

    df = download_raw_images(df, Path(args.output_dir))

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved metadata for {len(df)} photos to {args.output_csv}")

    print(f"\nSaved {len(df)} rows to {args.output_csv}")
    print(df[["observation_id", "latitude", "longitude",
              "observed_on", "local_path", "download_status"]].to_string())

    logging.info("iNaturalist observation fetch complete")


if __name__ == '__main__':
    main()