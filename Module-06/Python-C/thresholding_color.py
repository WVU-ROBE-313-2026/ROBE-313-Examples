import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load image in BGR and convert to RGB for accurate Matplotlib display
    img_bgr = cv2.imread('yellowtargets.png')
    
    if img_bgr is None:
        print("Error: Could not load 'yellowtargets.png'. Please check the file path.")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV color space for robust color segmentation
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for 'yellow' in OpenCV's HSV space.
    # Note: In OpenCV, Hue ranges from 0-179. Yellow is typically between 20 and 35.
    # Saturation and Value are set high to ignore washed-out pixels, whites, and shadows.
    lower_h = 20
    upper_h = 35
    lower_yellow = np.array([lower_h, 100, 100])
    upper_yellow = np.array([upper_h, 255, 255])

    # Create a binary mask where pixels falling within the yellow range become 255 (white)
    binary_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # Extract the Hue channel for visualization
    hue_channel = img_hsv[:, :, 0]

    # --- Display results in a 2x2 grid ---
    plt.figure(figsize=(10, 8))

    # 1. Original Color Image
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.title('Original Color (RGB)')
    plt.axis('off')

    # 2. Hue Channel
    # We use the 'hsv' colormap so the grayscale values map roughly to their visual colors
    plt.subplot(222)
    plt.imshow(hue_channel, cmap='hsv')
    plt.title('HSV: Hue Channel Only')
    plt.axis('off')

    # 3. Final Binary Mask
    plt.subplot(223)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Color Mask (Yellow)')
    plt.axis('off')

    # 4. Hue Histogram
    plt.subplot(224)
    # Plot histogram of the Hue channel (0 to 179 in OpenCV)
    plt.hist(hue_channel.ravel(), bins=180, range=(0, 180), color='gold', alpha=0.7)
    
    # Highlight the threshold range
    plt.axvline(lower_h, color='red', linestyle='dashed', linewidth=2, label=f'Lower Bound ({lower_h})')
    plt.axvline(upper_h, color='blue', linestyle='dashed', linewidth=2, label=f'Upper Bound ({upper_h})')
    plt.axvspan(lower_h, upper_h, color='yellow', alpha=0.3, label='Target Region')

    plt.title('Hue Channel Histogram')
    plt.xlabel('Hue Value [0, 179]')
    plt.ylabel('Pixel Count')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    