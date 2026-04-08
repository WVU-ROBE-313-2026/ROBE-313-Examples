import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_harris(img_path):
    """
    Loads an image, applies the Harris Corner Detector, and draws markers.
    Returns the annotated image (as 3-channel grayscale for Matplotlib).
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not load '{img_path}'.")
        return None

    # Convert to grayscale for the math, and keep a 3-channel gray for drawing
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Harris detector requires float32 input
    gray_float = np.float32(gray)

    # --- OpenCV Harris Corner Detector ---
    # Parameters: (image, blockSize, ksize, k)
    # blockSize: Size of neighborhood considered for corner detection
    # ksize: Aperture parameter for the Sobel derivative
    # k: Harris detector free parameter in the equation
    dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

    # Dilate the response map to merge adjacent corner pixels for cleaner drawing
    dst = cv2.dilate(dst, None)

    # Threshold the response map: keeping only responses greater than 10% of the max
    threshold = 0.1 * dst.max()
    
    # Get the (y, x) coordinates of the strong corners
    corners = np.argwhere(dst > threshold)
    
    # Draw small white squares with black borders (matching the slide)
    for y, x in corners:
        cv2.rectangle(display_img, (x-2, y-2), (x+2, y+2), (255, 255, 255), -1) # White fill
        cv2.rectangle(display_img, (x-2, y-2), (x+2, y+2), (0, 0, 0), 1)       # Black border

    return display_img

def main():
    # Process both views
    img1_annotated = apply_harris('building2-1.png')
    img2_annotated = apply_harris('building2-2.png')

    if img1_annotated is None or img2_annotated is None:
        return

    # Set up the 2x2 grid
    plt.figure(figsize=(12, 8))

    # --- View One ---
    plt.subplot(221)
    plt.imshow(img1_annotated)
    plt.title('(a) View One')
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')

    # --- Zoomed View One ---
    plt.subplot(222)
    plt.imshow(img1_annotated)
    plt.title('(b) Zoomed-in View One')
    # Use matplotlib limits to zoom in programmatically
    height, width, _ = img1_annotated.shape
    plt.xlim(width * 0.35, width * 0.75)
    plt.ylim(height * 0.45, height * 0.05) # Inverted Y for image coordinates
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')

    # --- View Two ---
    plt.subplot(223)
    plt.imshow(img2_annotated)
    plt.title('(c) View Two')
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')

    # --- Zoomed View Two ---
    plt.subplot(224)
    plt.imshow(img2_annotated)
    plt.title('(d) Zoomed-in View Two')
    # Apply similar zoom to the second image
    plt.xlim(width * 0.15, width * 0.55) # Shifted slightly left to track the building feature
    plt.ylim(height * 0.45, height * 0.05)
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()