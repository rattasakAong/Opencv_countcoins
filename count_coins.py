import cv2
import numpy as np

image = cv2.imread('money.jpg')
roi = image[:1080, 0:1920]

#convert color to gray
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# delete noise
gray_blur = cv2.GaussianBlur(gray, (15,15), 0)

# chang gray to binary
thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

#closing
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

# find contours
result_img = closing.copy()
contours, hierachy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# scan area and create ellipse
counter = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 5000 or area > 35000:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(roi, ellipse, (0, 255, 0), 3)
    counter+=1

cv2.putText(roi, f'{counter}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imshow('show', roi)


cv2.waitKey(0)
cv2.destroyAllWindows()
