import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("pic2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

threshold = 20
min_size = 32

def split(img, x, y, w, h, regions):
    region = img[y:y+h, x:x+w]
    
    if w <= min_size or h <= min_size:
        regions.append((x, y, w, h))
        return

    if np.std(region) < threshold:
        regions.append((x, y, w, h))
        return

    w2 = w // 2
    h2 = h // 2

    split(img, x, y, w2, h2, regions)
    split(img, x+w2, y, w-w2, h2, regions)
    split(img, x, y+h2, w2, h-h2, regions)
    split(img, x+w2, y+h2, w-w2, h-h2, regions)


regions = []
h, w, _ = img.shape

split(img, 0, 0, w, h, regions)

split_img = img.copy()

for i, (x, y, rw, rh) in enumerate(regions):
    cv2.rectangle(split_img, (x, y), (x+rw, y+rh), (255, 0, 0), 2)

merged_img = img.copy()

for x, y, rw, rh in regions:
    mean = np.mean(img[y:y+rh, x:x+rw], axis=(0, 1))
    merged_img[y:y+rh, x:x+rw] = mean

cv2.imwrite("region_splitting.png", cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR))
cv2.imwrite("region_merging.png", cv2.cvtColor(merged_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(split_img)
plt.title("Region Splitting")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(merged_img.astype(np.uint8))
plt.title("Region Merging")
plt.axis("off")

plt.show()