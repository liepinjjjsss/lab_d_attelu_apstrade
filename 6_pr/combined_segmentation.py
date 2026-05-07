import numpy as np
import matplotlib.pyplot as plt
import cv2

images = [
    { 
        # slightly cloudy, slightly opaque thin coulds over the sea
        "true":  "6_pr/source_images/2026_04_01/2026-04-01-00_00_2026-04-01-23_59_Sentinel-2_L2A_True_color.jpg",
        "false": "6_pr/source_images/2026_04_01/2026-04-01-00_00_2026-04-01-23_59_Sentinel-2_L2A_False_color.jpg",
        "label": "2026-04-01",
        "threshold": 30,
        "min_area": 50,
        "kernel_open":  2,
        "kernel_close": 10,
    },
    {
        # many thick standalone clouds that cast shadows on the ground
        "true":  "6_pr/source_images/2026_04_03/2026-04-03-00_00_2026-04-03-23_59_Sentinel-2_L2A_True_color.jpg",
        "false": "6_pr/source_images/2026_04_03/2026-04-03-00_00_2026-04-03-23_59_Sentinel-2_L2A_False_color(1).jpg",
        "label": "2026-04-03",
        "threshold": 15,
        "min_area": 50,
        "kernel_open":  3,
        "kernel_close": 10,
    },
    {
        # completely clear weather, not a single cloud
        "true":  "6_pr/source_images/2026_04_23/2026-04-23-00_00_2026-04-23-23_59_Sentinel-2_L2A_True_color(1).jpg",
        "false": "6_pr/source_images/2026_04_23/2026-04-23-00_00_2026-04-23-23_59_Sentinel-2_L2A_False_color.jpg",
        "label": "2026-04-23",
        "threshold": 30,
        "min_area": 30,
        "kernel_open":  2,
        "kernel_close": 8,
    },
]

def process_image(false_img, threshold, min_area, kernel_open, kernel_close):
    b, g, r = cv2.split(false_img)

    _, water_mask = cv2.threshold(r, threshold, 255, cv2.THRESH_BINARY_INV)

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open,  kernel_open))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
    mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask,       cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    overlay = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB).copy()
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            overlay[labels == label] = [0, 255, 50]

    return overlay, mask


for scene in images:
    true_img  = cv2.imread(scene["true"])
    false_img = cv2.imread(scene["false"])

    overlay, mask = process_image(
        false_img,
        scene["threshold"],
        scene["min_area"],
        scene["kernel_open"],
        scene["kernel_close"],
    )

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(scene["label"], fontsize=13, fontweight="bold")

    axes[0].imshow(cv2.cvtColor(true_img,  cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(false_img, cv2.COLOR_BGR2RGB)) 
    axes[2].imshow(mask, cmap="gray")                          
    axes[3].imshow(overlay)      

    axes[0].set_title("true color")
    axes[1].set_title("false color (NIR)")        
    axes[2].set_title("mask")
    axes[3].set_title("result")                    

    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"{scene['label']} — parametri: threshold={scene['threshold']}, "
          f"min_area={scene['min_area']}, "
          f"kernel_open={scene['kernel_open']}, kernel_close={scene['kernel_close']}")
