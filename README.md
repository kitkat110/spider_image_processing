# Spider Image Processing Pipeline

A pipeline for segmenting *Gasteracantha cancriformis* (Spinyback Orbweaver) images and extracting morphological color features from the abdomen and spines. Images are sourced directly from [iNaturalist](https://www.inaturalist.org/) and processed using deep learning-based background removal (U2Net) followed by region separation and color feature extraction.

---

## Overview

```
get_observations.py  →  data/observations.csv + data/observations/
                                    ↓
              pipeline.py  →  data/results.csv + data/segmented/
```

The workflow is split into two stages:

1. **`get_observations.py`** — fetches research-grade observations from iNaturalist, downloads raw images, and saves metadata (including GPS coordinates) to a CSV.
2. **`pipeline.py`** — reads that CSV, runs each image through preprocessing, segmentation, and color feature extraction, and writes all results back into a single DataFrame alongside the original location and photo metadata.

---

## Project Structure

```
spider_image_processing/
├── get_observations.py       # Fetch + download raw iNaturalist observations
├── pipeline.py               # Full processing pipeline over downloaded images
├── test_single_image.py      # Test script for verifying pipeline on one image
│
├── src/
│   ├── preprocessing.py      # Image loading, resizing, CLAHE enhancement
│   ├── u2net_segmentation.py # U2Net model loading + spider mask generation
│   ├── region_separation.py  # Abdomen and spine mask separation
│   ├── color_classification.py # HSV color classification by reference distance
│   └── features.py           # Abdomen and spine color feature extraction
│
├── model/
│   ├── __init__.py
│   └── u2net.py              # U2Net / U2Net-P model architecture
│
├── tests/                    # Unit tests
├── saved_models/             # U2Net-P weights (not tracked by Git)
│   └── u2netp/
│       └── u2netp.pth
└── data/                     # Output directory (created at runtime, not tracked by Git)
    ├── observations/         # Raw downloaded PNGs
    ├── segmented/            # Segmented output PNGs
    ├── observations.csv      # Metadata from get_observations.py
    └── results.csv           # Full results from pipeline.py
```

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install opencv-python numpy pandas requests pillow torch torchvision
```

**Model weights** — download U2Net-P weights and place them at:
```
saved_models/u2netp/u2netp.pth
```
Weights are available from the [U2Net repository](https://github.com/xuebinqin/U2-Net).

---

## Usage

### Step 1 — Fetch observations from iNaturalist

Downloads research-grade *G. cancriformis* observations and saves one raw image per observation locally.

```bash
python get_observations.py --count 50 --output_dir data/observations --output_csv data/observations.csv
```

| Argument | Default | Description |
|---|---|---|
| `--count` | `20` | Number of observations to fetch |
| `--output_dir` | `data/observations` | Directory to save raw PNGs |
| `--output_csv` | `data/observations.csv` | Path for the metadata CSV |
| `-l` / `--loglevel` | `WARNING` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

The output CSV includes: `observation_id`, `photo_id`, `latitude`, `longitude`, `place_guess`, `observed_on`, `quality_grade`, `user_login`, `taxon_name`, `common_name`, `local_path`, `download_status`.

### Step 2 — Run the pipeline

Preprocesses each raw image and runs it through segmentation and color extraction. All results are merged back into the observations DataFrame.

```bash
python pipeline.py --input_csv data/observations.csv --output_csv data/results.csv
```

| Argument | Default | Description |
|---|---|---|
| `--input_csv` | `data/observations.csv` | CSV from Step 1 |
| `--output_csv` | `data/results.csv` | Where to save results |
| `--segmented_dir` | `data/segmented` | Directory to save segmented PNGs |
| `--model_path` | `saved_models/u2netp/u2netp.pth` | Path to U2Net-P weights |
| `--device` | `cpu` | `cpu` or `cuda` |
| `-l` / `--loglevel` | `WARNING` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

The output CSV contains all original metadata columns plus: `abdomen_color`, `spine_color`, `segmented_image_path`, and `processing_status`. Mask arrays are used during processing but excluded from the CSV.

---

## How It Works

**Preprocessing** (`src/preprocessing.py`)
Raw images are resized to 512×512 and contrast-enhanced using CLAHE on the L-channel in LAB color space, normalizing lighting variation across field photos.

**Segmentation** (`src/u2net_segmentation.py`)
A pretrained U2Net-P model generates a binary foreground mask isolating the spider from the background. Morphological closing fills small holes, and only the largest connected component is kept.

**Region separation** (`src/region_separation.py`)
The spider mask is split into abdomen and spine sub-regions. Otsu's thresholding on the brightness channel identifies the abdomen, and red HSV ranges combined with spatial filtering identify the spines.

**Color classification** (`src/color_classification.py`, `src/features.py`)
Mean HSV values are extracted from each masked region and matched to a set of reference colors using weighted Euclidean distance. Abdomen color is classified from the full reference set; spine color is classified as red or black based on the fraction of red pixels in the spine mask.

---

## Data Source

Observations are fetched from the [iNaturalist API](https://api.inaturalist.org/v1/docs/) filtered to:
- Taxon: *Gasteracantha cancriformis* (taxon ID `57486`)
- Quality grade: `research` (community-validated identifications only)
- One photo per observation (the observer's featured photo)

All observation data is subject to [iNaturalist's terms of use](https://www.inaturalist.org/pages/terms).
