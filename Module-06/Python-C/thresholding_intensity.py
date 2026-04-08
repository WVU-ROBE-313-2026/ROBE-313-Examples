import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('castle.png', cv2.IMREAD_GRAYSCALE)
img = img / 255.0  # Normalize to [0,1]

# Otsu's thresholding
otsu_threshold_value, binary_otsu = cv2.threshold(
    (img*255).astype(np.uint8), 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

otsu_norm = otsu_threshold_value / 255.0
print("Otsu's threshold value:", otsu_norm)

# Manual threshold (t = 0.7)
manual_t = 0.3
binary_manual = (img >= manual_t).astype(np.uint8) * 255

# Display results in a 2x2 grid
plt.figure(figsize=(10, 8))

# 1. Original Image
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

# 2. Otsu Threshold Result
plt.subplot(222)
plt.imshow(binary_otsu, cmap='gray')
plt.title("Otsu (t={:.2f})".format(otsu_norm))
plt.axis('off')

# 3. Manual Threshold Result
plt.subplot(223)
plt.imshow(binary_manual, cmap='gray')
plt.title(f'Manual (t={manual_t})')
plt.axis('off')

# 4. Normalized Histogram
plt.subplot(224)
# Flatten the normalized image matrix to a 1D array for the histogram
plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), color='gray', alpha=0.7)

# Add vertical lines to show where the thresholds cut the distribution
plt.axvline(otsu_norm, color='red', linestyle='dashed', linewidth=2, label=f'Otsu (t={otsu_norm:.2f})')
plt.axvline(manual_t, color='blue', linestyle='dashed', linewidth=2, label=f'Manual (t={manual_t})')

plt.title('Normalized Intensity Histogram')
plt.xlabel('Intensity [0, 1]')
plt.ylabel('Pixel Count')
plt.legend()

plt.tight_layout()
plt.show()
