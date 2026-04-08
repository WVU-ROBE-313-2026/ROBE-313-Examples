import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the grayscale image
img = cv2.imread('street.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load street.png. Check file path.")
    exit()

# 2. Simulate Exposure Changes
# cv2.convertScaleAbs calculates: dst = saturate_cast<uchar>(|src * alpha + beta|)
# Alpha < 1 reduces contrast/brightness (Underexposed)
img_under = cv2.convertScaleAbs(img, alpha=0.4, beta=0)

# Alpha > 1 pushes values past 255, which clip to white (Overexposed)
img_over = cv2.convertScaleAbs(img, alpha=1.1, beta=30)

# 3. Setup Matplotlib Figure (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.tight_layout(pad=4.0)

images = [
    (img, "Original Image"),
    (img_under, "Underexposed (Shifted Left)"),
    (img_over, "Overexposed (Clipped Right)")
]

# 4. Plot Images and their Histograms
for i, (image, title) in enumerate(images):
    # Plot Image
    ax_img = axes[i, 0]
    ax_img.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax_img.set_title(title)
    ax_img.axis('off')
    
    # Plot Histogram
    ax_hist = axes[i, 1]
    # Use .ravel() to flatten, 256 bins, range 0-256
    ax_hist.hist(image.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
    ax_hist.set_title(f"{title} - Histogram")
    ax_hist.set_xlim([0, 256])
    ax_hist.set_xlabel('Pixel Intensity')
    ax_hist.set_ylabel('Pixel Count')
    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
