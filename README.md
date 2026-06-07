# Spider Image Processing Pipeline

A pipeline for segmenting *Gasteracantha cancriformis* (Spinyback Orbweaver) images and extracting morphological color features from the abdomen and spines. Images are sourced directly from [iNaturalist](https://www.inaturalist.org/) and processed using deep learning-based background removal (U2Net) followed by region separation and color feature extraction.

---

## Overview

```
iNaturalist API → get_observations.py → data/observations.csv + images/
                                              ↓
                               pipeline.py → data/results.csv
```

The workflow is split into two stages:

1. **`get_observations.py`** — fetches research-grade observations from iNaturalist, downloads and preprocesses images, and saves metadata (including GPS coordinates) to a CSV
2. **`pipeline.py`** — reads that CSV and runs each image through segmentation and color feature extraction

A single-image entry point (`test_single_image.py`) is also provided for local images.

---

## Project Structure

```
spider_image_processing/
├── get_observations.py       # Fetch + download iNaturalist observations
├── pipeline.py               # Run segmentation pipeline over downloaded images
│
├── src/
│   ├── preprocessing.py      # Image loading, resizing, CLAHE enhancement
│   ├── u2net_segmentation.py # U2Net model loading + spider mask generation
│   ├── region_separation.py  # Abdomen and spine mask separation
│   └── features.py           # Color feature extraction
│
├── model/
│   ├── __init__.py
│   └── u2net.py              # U2Net / U2Net-P model architecture
│
├── utils/
│   └── logging_config.py     # Shared logging setup
│
├── tests/                    # Unit tests
├── saved_models/             # U2Net-P weights (not tracked by Git)
│   └── u2netp/
│       └── u2netp.pth
└── data/                     # Output directory (created at runtime)
    ├── observations/         # Downloaded + preprocessed PNGs
    ├── observations.csv      # Metadata from get_observations.py
    └── results.csv           # Color features from inaturalist_pipeline.py
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

Downloads research-grade *G. cancriformis* observations and saves preprocessed images locally.

```bash
python get_observations.py --count 50 --output_dir data/observations --output_csv data/observations.csv
```

| Argument | Default | Description |
|---|---|---|
| `--count` | `20` | Number of photos to fetch |
| `--output_dir` | `data/observations` | Directory to save downloaded PNGs |
| `--output_csv` | `data/observations.csv` | Path for the metadata CSV |
| `--loglevel` | `WARNING` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |

The output CSV includes: `observation_id`, `photo_id`, `latitude`, `longitude`, `place_guess`, `observed_on`, `quality_grade`, `user_login`, `local_path`, `download_status`.

### Step 2 — Run the pipeline

Runs segmentation and color extraction over all successfully downloaded images.

```bash
python inaturalist_pipeline.py --input_csv data/observations.csv --output_csv data/results.csv
```

| Argument | Default | Description |
|---|---|---|
| `--input_csv` | `data/observations.csv` | CSV from Step 1 |
| `--output_csv` | `data/results.csv` | Where to save results |
| `--model_path` | `saved_models/u2netp/u2netp.pth` | Path to U2Net-P weights |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--loglevel` | `WARNING` | Logging verbosity |

The results CSV appends `abdomen_color`, `spine_color`, and `processing_status` to all original metadata columns.

### Single image (local file)

```bash
python pipeline.py --image path/to/spider.jpg --device cpu
```

---

## How It Works

**Preprocessing** (`src/preprocessing.py`)
Images are resized to 512×512 and contrast-enhanced using CLAHE on the L-channel in LAB color space. This normalises lighting variation across field photos.

**Segmentation** (`src/u2net_segmentation.py`)
A pretrained U2Net-P model generates a binary foreground mask, isolating the spider from the background.

**Region separation** (`src/region_separation.py`)
The spider mask is further split into abdomen and spine sub-regions for independent feature extraction.

**Feature extraction** (`src/features.py`)
Mean color values are extracted from each masked region and returned as `abdomen_color` and `spine_color`.

---

## Data Source

Observations are fetched from the [iNaturalist API](https://api.inaturalist.org/v1/docs/) filtered to:
- Taxon: *Gasteracantha cancriformis* (taxon ID `57486`)
- Quality grade: `research` (community-validated identifications only)

All observation data is subject to [iNaturalist's terms of use](https://www.inaturalist.org/pages/terms).