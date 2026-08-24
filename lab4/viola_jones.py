import sys

# Use the temporary OpenCV 4 installation
sys.path.insert(0, r".\opencv4_temp")

import cv2
import os

# ==============================
# INPUT
# ==============================
image_path = "pic6.jpeg"

# ==============================
# LOAD IMAGE
# ==============================
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load pic6.jpeg")
    exit()

print("Image loaded successfully.")
print("Image size:", image.shape)

# ==============================
# LOAD VIOLA-JONES HAAR CASCADE
# ==============================
cascade_path = os.path.join(
    os.path.dirname(cv2.__file__),
    "data",
    "haarcascade_frontalface_default.xml"
)

print("Cascade:", cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Could not load Haar cascade.")
    exit()

print("Haar cascade loaded successfully.")

# ==============================
# GRAYSCALE
# ==============================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==============================
# VIOLA-JONES DETECTION
# ==============================
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(15, 15)
)

print("Faces detected:", len(faces))

# ==============================
# DRAW BOUNDING BOXES
# ==============================
for i, (x, y, w, h) in enumerate(faces):

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Face {i + 1}",
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

# ==============================
# SAVE
# ==============================
output_path = "viola_jones_output.jpg"

cv2.imwrite(output_path, image)

print("Output saved to:", output_path)

# ==============================
# DISPLAY
# ==============================
cv2.imshow("Viola-Jones Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()