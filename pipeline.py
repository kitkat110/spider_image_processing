from preprocessing import load_images
from u2net_segmentation import segment_spider_u2net
from region_separation import sep_abdomen_spine, get_segmented_spider_image
from features import extract_abdomen_color, extract_spine_color

def process_spider_image(img_path, u2net_model, device="cpu"):
    image = load_images(img_path)
    spider_mask = segment_spider_u2net(image, u2net_model, device=device)
    abdomen_mask, spine_mask = sep_abdomen_spine(image, spider_mask)
    abdomen_color = extract_abdomen_color(image, abdomen_mask)
    spine_color = extract_spine_color(image, spine_mask)
    segmented_image = get_segmented_spider_image(image, spider_mask)

    return {
        "abdomen_color": abdomen_color,
        "spine_color": spine_color,
        "segmented_image": segmented_image,
        "abdomen_mask": abdomen_mask,
        "spine_mask": spine_mask,
        "spider_mask": spider_mask
    }