# Spider Image Processing Pipeline

## About 
This project provides a pipeline for segmenting spiders from images and extracting
abdomen and spine colors using image processing and deep learning (U2NETP).

## Project Structure

```
spider_image_processing/
├── config.py # Paths and constants
├── preprocessing.py # Image preprocessing functions
├── u2net_segmentation.py # U2NETP model loading and spider segmentation
├── region_separation.py # Separate abdomen and spine masks
├── color_classification.py # Color classification logic
├── features.py # Functions for extracting color features
├── pipeline.py # Main pipeline to process a single image
├── test_single_image.py # Run a test image
├── model/
│ ├── init.py
│ └── u2net.py # U2NETP model definition
├── saved_models/ # Store pre-trained U2NETP weights (ignored by Git)
├── .gitignore # Files to ignore in Git
└── README.md
```
