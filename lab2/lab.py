import numpy as np
import matplotlib.pyplot as plt
import cv2

def region_growing(image, seed_point, threshold):
    rows, cols = image.shape
    region = np.zeros((rows, cols), dtype=bool)
    visited = np.zeros((rows, cols), dtype=bool)
    intensity = image[seed_point]
    
    region[seed_point] = True
    visited[seed_point] = True
    to_process = [seed_point]
    
    while to_process:
        current_point = to_process.pop()
        r, c = current_point
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                if abs(int(image[nr, nc]) - int(intensity)) <= threshold:
                    region[nr, nc] = True
                    to_process.append((nr, nc))
    
    return region

#def region_merging():
    

if __name__ == "__main__":
    image = cv2.imread("pic2.jpg", cv2.IMREAD_GRAYSCALE)

    seed_point = (5, 5)
    threshold = 70
    segmented_region = region_growing(image, seed_point, threshold)
    output = np.zeros_like(image)
    output[segmented_region] = image[segmented_region]

    cv2.imwrite("region_growing_output.png", output)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray", interpolation="none")
    plt.scatter(seed_point[1], seed_point[0], color="red")  
    
    plt.subplot(1, 2, 2)
    plt.title("Segmented Region")
    plt.imshow(segmented_region, cmap="gray", interpolation="none")
    plt.show()
    