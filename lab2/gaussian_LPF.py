import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("pic3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply Gaussian Low Pass Filter
filtered = cv2.GaussianBlur(img, (5, 5), 0)

# Save images
cv2.imwrite("original.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite("low_pass_filtered.png", cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))

# Display
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(122)
plt.imshow(filtered)
plt.title("Low Pass Filter")
plt.axis("off")

plt.tight_layout()
plt.show()