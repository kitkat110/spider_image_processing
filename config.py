import os

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

U2NET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "u2netp",
    "u2netp.pth"
)