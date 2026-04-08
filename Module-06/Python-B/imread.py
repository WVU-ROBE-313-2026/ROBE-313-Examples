import cv2
import matplotlib.pyplot as plt

# Load image (OpenCV uses BGR by default)
img = cv2.imread('flowers8.png')

# Convert BGR to RGB for matplotlib plotting
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img)
plt.title('Street Scene (BGR)')
plt.axis('off')
plt.show()

plt.imshow(img_rgb)
plt.title('Street Scene (RGB)')
plt.axis('off')
plt.show()

# Understand the underlying numpy array
print(f"Shape: {img.shape} -> (Height, Width, Channels)")
