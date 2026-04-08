import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_rotated_square_edges(size, rotation_angle=40):
    """
    Dynamically generates a binary edge map of a single rotated square.
    """
    # Create blank black image
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Define square vertices (centered)
    s_size = size // 4
    center = size // 2
    
    # Vertices: [x1, y1], [x2, y2], [x3, y3], [x4, y4]
    vertices = np.array([
        [center - s_size, center - s_size],
        [center + s_size, center - s_size],
        [center + s_size, center + s_size],
        [center - s_size, center + s_size]
    ], dtype=np.int32)
    
    # Apply rotation
    M = cv2.getRotationMatrix2D((center, center), rotation_angle, 1.0)
    
    # Rotate the polygon points
    ones = np.ones(shape=(len(vertices), 1))
    points_ones = np.hstack([vertices, ones])
    rotated_vertices = M.dot(points_ones.T).T.astype(np.int32)
    
    # Draw only the outline of the square in white on black background
    cv2.polylines(img, [rotated_vertices], True, (255), 2)
    
    # The output is directly the binary edge map that Hough requires
    return img

def main():
    # 1. GENERATE INPUT (Stage A)
    # Generate the binary edge image of the square outline
    img_size = 500
    edge_map = generate_rotated_square_edges(img_size, rotation_angle=40)

    # 2. COMPUTE HOUGH TRANSFORM (Stage B)
    # cv2.HoughLines: parameters (image, rho_precision, theta_precision, threshold)
    # theta_precision (1 degree) = np.pi/180
    lines = cv2.HoughLines(edge_map, 1, np.pi/180, 200)

    # Note: cv2.HoughLines does not return the full accumulator matrix needed 
    # for visualization (stage B). To visualize the dense Hough Space accurately, 
    # we would need to implement the voting math manually, which is slow in Python.
    # For this demo, we can show the extracted lines over the accumulator concept.
    
    # 3. PEAK SELECTION (Stage C: conceptual)
    # cv2.HoughLines automatically sorts the lines by their 'votes',
    # so the top 4 are already our 4 peaks.
    if lines is not None and len(lines) >= 4:
        top_4_lines = lines[:4]
    else:
        top_4_lines = lines if lines is not None else []
        print("Error: Could not find 4 dominant lines.")

    # 4. RECONSTRUCT LINES (Stage D)
    # Create a color image to draw the output
    output_img = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
    
    line_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)] # G, B, R, Y
    
    print("\n--- Extracted Lines (Rho, Theta) ---")
    
    for i in range(len(top_4_lines)):
        rho, theta = top_4_lines[i][0]
        color = line_colors[i]
        
        # Convert polar (rho, theta) to Cartesian (x, y) infinite lines for cv2.line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Calculate two far-off points to define the infinite line
        line_length_factor = img_size * 2
        x1 = int(x0 + line_length_factor * (-b))
        y1 = int(y0 + line_length_factor * (a))
        x2 = int(x0 - line_length_factor * (-b))
        y2 = int(y0 - line_length_factor * (a))
        
        # Draw the long line on the image
        cv2.line(output_img, (x1, y1), (x2, y2), color, 2)
        
        # Print the data that the robot controller receives
        print(f"Line {i+1}: ({rho:.2f}, {theta:.2f}) [Vote count: {top_4_lines[i][0][0]}]")
    
    # Display the final results as a 1x2 comparison
    plt.figure(figsize=(10, 5))
    
    # Original edge map
    plt.subplot(121)
    plt.imshow(edge_map, cmap='gray')
    plt.title('(a) Synthetic Binary Edge Map')
    plt.axis('off')
    
    # Final Output
    plt.subplot(122)
    plt.imshow(output_img)
    plt.title('(d) Reconstructed Dominant Lines')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
