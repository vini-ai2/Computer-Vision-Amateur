# spatial filtering
# - simple and weighted avg for 3x3, 5x5, 7x7 to observe the blurring
# use laplacian, sobel, canny filters
# sharpening effect using prewitt
# grey scale image for ede detection
# conclude which is better for edges
# perform geometric transformations 


#all imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("output", exist_ok=True)


# creating a fn to automatically display the before and after images


save_count = 1

def display(before, after, title1="Before", title2="After"):
    global save_count

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    if len(before.shape) == 2:
        plt.imshow(before, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if len(after.shape) == 2:
        plt.imshow(after, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))

    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"output/{save_count:02d}_{title2.replace(' ', '_')}.png")
    plt.close()

    save_count += 1

#Linear filtering 

image = cv2.imread('pic1.jpg')
print(image)
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout()
plt.savefig("output/00_Original_Image.png")
plt.close()

#moving on to the simple average filter
simple_avg3 = cv2.blur(image, (3, 3))
display(image, simple_avg3, "Original Image (3x3)", "Simple Average Filtered Image")

simple_avg5 = cv2.blur(image, (5, 5))
display(image, simple_avg5, "Original Image (5x5)", "Simple Average Filtered Image")

simple_avg7 = cv2.blur(image, (7, 7))
display(image, simple_avg7, "Original Image (7x7)", "Simple Average Filtered Image")

#moving to weighted average filter
weighted_avg3 = cv2.GaussianBlur(image, (3, 3), 0)
display(image, weighted_avg3, "Original Image (3x3)", "Weighted Average Filtered Image")

weighted_avg5 = cv2.GaussianBlur(image, (5, 5), 0)
display(image, weighted_avg5, "Original Image (5x5)", "Weighted Average Filtered Image")

weighted_avg7 = cv2.GaussianBlur(image, (7, 7), 0)
display(image, weighted_avg7, "Original Image (7x7)", "Weighted Average Filtered Image")

# grey scale image for edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# using laplacitan filter for edge detection
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
display(gray_image, laplacian, "Grey Scale Image", "Laplacian Filtered Image")

# using sobel filter for edge detection
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobelx = cv2.convertScaleAbs(sobelx)
display(gray_image, sobelx, "Grey Scale Image", "Sobel Filtered Image (X direction)")

sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
sobely = cv2.convertScaleAbs(sobely)
display(gray_image, sobely, "Grey Scale Image", "Sobel Filtered Image (Y direction)")

sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
display(gray_image, sobelxy, "Grey Scale Image", "Sobel Filtered Image (Both directions)")

# using canny filter for edge detection
canny = cv2.Canny(gray_image, 30, 100)
display(gray_image, canny, "Grey Scale Image", "Canny Filtered Image")

# sharpening effect using prewitt filter
# Prewitt kernels
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])

prewitt_x_filtered = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_x)
prewitt_x_filtered = cv2.convertScaleAbs(prewitt_x_filtered)
display(gray_image, prewitt_x_filtered, "Grey Scale Image", "Prewitt Filtered Image (X direction)")

prewitt_y_filtered = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_y)
prewitt_y_filtered = cv2.convertScaleAbs(prewitt_y_filtered)
display(gray_image, prewitt_y_filtered, "Grey Scale Image", "Prewitt Filtered Image (Y direction)")

# perform geometric transformations

rows, cols = image.shape[:2]

# translation
translation_matrix = np.float32([[1, 0, 100],
                                 [0, 1, 50]])
translated = cv2.warpAffine(image, translation_matrix, (cols, rows))
display(image, translated, "Original Image", "Translated Image")

# rotation
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
display(image, rotated, "Original Image", "Rotated Image")

# scaling
scaled = cv2.resize(image, None, fx=3, fy=5, interpolation=cv2.INTER_LINEAR)
display(image, scaled, "Original Image", "Scaled Image")

# horizontal flip
horizontal_flip = cv2.flip(image, 1)
display(image, horizontal_flip, "Original Image", "Horizontal Flip")

# vertical flip
vertical_flip = cv2.flip(image, 0)
display(image, vertical_flip, "Original Image", "Vertical Flip")

# affine transformation
pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])

pts2 = np.float32([[10, 100],
                   [200, 50],
                   [100, 250]])

affine_matrix = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(image, affine_matrix, (cols, rows))
display(image, affine, "Original Image", "Affine Transformation")

