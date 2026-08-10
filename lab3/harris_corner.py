import cv2 
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('pic5.jpg')
opImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
opImg = np.float32(opImg)

dest = cv2.cornerHarris(opImg, 2, 3, 0.01) #image, neighourhood size, sobel kernel, harris free parameter
dest = cv2.dilate(dest, None) #dilation helps in detecting the corners better

#image[dest>0.1*dest.max()] = [0, 0, 255] #corners greater than the given threshold are marked red
#to mark the corners with circles
coordinates = np.where(dest>0.2*dest.max())
for y, x in zip (*coordinates):
    cv2.circle(image, (x, y), 3, (0, 0, 255), 1)
cv2.imwrite("harris_corners.jpeg",image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()