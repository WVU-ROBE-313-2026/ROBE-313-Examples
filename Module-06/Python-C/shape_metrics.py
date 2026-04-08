import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load image in grayscale
    # Make sure 'shark1.png' is in the same directory
    img_gray = cv2.imread('shark1.png', cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        print("Error: Could not load 'shark1.png'.")
        return

    # 1. Binarize the image (assuming a light object on a dark background)
    _, binary_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # 2. Find connected components (Contours in OpenCV)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No objects found!")
        return

    # Grab the largest blob (assuming our shark is the main object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a color image to draw our colorful annotations on
    output_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)

    # --- A. IMAGE MOMENTS & CENTROID ---
    # Calculate moments (m00, m10, m01, etc.)
    M = cv2.moments(largest_contour)
    
    # Calculate centroid (u, v) using spatial moments
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
        
    # Draw the centroid as a red dot
    cv2.circle(output_img, (cx, cy), 5, (255, 0, 0), -1) 

    # --- B. BOUNDING BOX ---
    # Calculate the axis-aligned enclosing rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Draw the bounding box in green
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- C. EQUIVALENT ELLIPSE ---
    # Fit an ellipse to the contour (requires at least 5 points)
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        # Draw the ellipse in blue
        cv2.ellipse(output_img, ellipse, (0, 0, 255), 2)
        
        # Extract orientation angle from the ellipse tuple for the console output
        (center_x, center_y), (axes_width, axes_height), angle = ellipse
    else:
        angle = 0.0

    # Print the calculated robotics metrics to the console
    print("--- Shape Metrics Extracted ---")
    print(f"Area (m00):      {M['m00']} pixels")
    print(f"Centroid (u, v): ({cx}, {cy})")
    print(f"Orientation:     {angle:.2f} degrees")
    print("-------------------------------")

    # Display the result using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(output_img)
    plt.title("Extracted Features: Bounding Box, Ellipse, & Centroid")
    plt.xlabel("u (pixels)")
    plt.ylabel("v (pixels)")
    
    # Invert Y axis to match standard image coordinate systems (origin top-left)
    plt.gca().invert_yaxis() 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
