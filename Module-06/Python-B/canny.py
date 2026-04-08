import cv2
import numpy as np

# Load an image
img = cv2.imread('street.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found. Please ensure 'image.jpg' is in the same directory.")
else:
    # Apply Canny edge detection
    # Parameters:
    # 1. Input image (8-bit grayscale)
    # 2. Lower threshold for hysteresis
    # 3. Upper threshold for hysteresis
    # 4. Aperture size for the Sobel operator (default is 3, can be 3, 5, or 7)
    # 5. L2gradient: Boolean flag indicating whether to use L2-norm for gradient magnitude
    edges = cv2.Canny(img, 100, 200, apertureSize=3, L2gradient=True)

    # Display the original and edge-detected images
    cv2.imshow('Original Image', img)
    cv2.imshow('Canny Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Conceptual Sobel Kernels (not directly used in Canny() function call)
    print("Conceptual Sobel X Kernel (apertureSize=3):")
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.int32)
    print(sobel_x_kernel)

    print("\nConceptual Sobel Y Kernel (apertureSize=3):")
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.int32)
    print(sobel_y_kernel)

