from imutils.perspective import four_point_transform
import cv2
import numpy

# Load image, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread("1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Find contours and sort for largest contour
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
print(cnts)
displayCnt = None

for c in cnts:
    # Perform contour approximation
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        displayCnt = approx
        break

# Obtain birds' eye view of image
warped = four_point_transform(image, displayCnt.reshape(4, 2))

cv2.imwrite("2.jpg", thresh)
cv2.imwrite("3.jpg", warped)

