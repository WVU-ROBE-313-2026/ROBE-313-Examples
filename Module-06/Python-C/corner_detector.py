
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load image in grayscale
    # Make sure 'castle.png' is in the same directory as this script
    img = cv2.imread('castle.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load 'castle.png'. Please check the file path.")
        return

    # Normalize to [0,1] for the manual thresholding math
    img_normalized = img / 255.0  

    # Otsu's thresholding (requires standard 0-255 uint8 format)
    # The '_' variable captures the optimal threshold value found by Otsu
    otsu_thresh_val, binary_otsu = cv2.threshold(
        img, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Manual threshold (t = 0.7) applied to the normalized image
    binary_manual = (img_normalized >= 0.7).astype(np.uint8) * 255

    # Display results side-by-side using matplotlib
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(binary_otsu, cmap='gray')
    plt.title(f"Otsu's Method (t={otsu_thresh_val/255:.2f})")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(binary_manual, cmap='gray')
    plt.title('Manual Threshold (t=0.70)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
