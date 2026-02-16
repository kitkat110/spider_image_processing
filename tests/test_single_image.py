import os
import sys
import matplotlib.pyplot as plt
from src.config import U2NET_MODEL_PATH
from src.u2net_segmentation import load_u2net_model
from src.pipeline import process_spider_image

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

device = "cpu"
u2net_model = load_u2net_model(U2NET_MODEL_PATH, device)

img_path = "/Users/minimal_kitkat/Desktop/spiny_orb4.jpg"
results = process_spider_image(img_path, u2net_model, device=device)

print("Abdomen Color:", results["abdomen_color"])
print("Spine Color:", results["spine_color"])

# Visualization 
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.title("Segmented Spider")
plt.imshow(results["segmented_image"])
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Abdomen Mask")
plt.imshow(results["abdomen_mask"], cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Spine Mask")
plt.imshow(results["spine_mask"], cmap='gray')
plt.axis('off')
plt.show()