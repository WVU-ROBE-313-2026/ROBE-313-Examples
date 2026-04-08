import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(img):
    """Pipeline: Gray, Blur, Canny."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(img):
    """Mask image to isolate the road area."""
    height, width = img.shape
    # Define vertices for a simple trapezoidal mask
    vertices = np.array([[(100, height), (width * 0.45, height * 0.6), (width * 0.55, height * 0.6), (width - 100, height)]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    """Draw line segments on the image."""
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img

# --- Main Demo Script ---
# 1. Load the real-world road lane image
# User: Replace 'lane.jpg' with the actual file path
img_path = 'lane.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from '{img_path}'")
    exit()

# Setup matplotlib figure
plt.figure(figsize=(12, 6))

# Process the image: preprocessing and ROI masking
canny = preprocess_image(img)
roi_canny = region_of_interest(canny)

# --- Scene 1: Illustrating Noise in the Lines ---
# We run Standard Hough Transform with a very low threshold to generate noise (many false positives)
lines_noisy = cv2.HoughLines(roi_canny, 1, np.pi/180, 50) # Extremely low threshold

noise_img = np.zeros_like(img)
if lines_noisy is not None:
    for line in lines_noisy:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(noise_img, (x1, y1), (x2, y2), (0, 0, 255), 1) # Draw in red for noise concept

plt.subplot(121)
plt.imshow(cv2.cvtColor(cv2.bitwise_or(img, noise_img), cv2.COLOR_BGR2RGB))
plt.title("1. Illustration of Noise / Uncleaned Detections")
plt.axis('off')

# --- Scene 2: Hough Transform Correcting Noise ---
# We apply the Probabilistic Hough Transform (more robust for real world lanes) to isolate key segments
# params: roi_canny, rho, theta, threshold, min_line_len, max_line_gap
lines_corrected = cv2.HoughLinesP(roi_canny, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)
line_img_corrected = draw_lines(img, lines_corrected)

# Blend with original
plt.subplot(122)
plt.imshow(cv2.cvtColor(cv2.addWeighted(img, 0.8, line_img_corrected, 1, 0), cv2.COLOR_BGR2RGB))
plt.title("2. Cleaned Detections: Hough Invariance Correcting Noise")
plt.axis('off')

plt.tight_layout()
plt.show()
