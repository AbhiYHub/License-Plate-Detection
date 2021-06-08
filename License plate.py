import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr

img = cv2.imread("image full path if in different location")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()

edged = cv2.Canny(gray, 30, 200)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.show()

contours, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

contour_with_plate = None
license_plate = None
x = None
y = None
w = None
h = None

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    if len(approx) == 4:
        contour_with_plate = approx
        xx, yy, w, h = cv2.boundingRect(contour)
        license_plate = gray[y:yy+h, x:xx+w]
        break

license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
(thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [contour_with_plate], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

(x, y) = np.where(mask == 255)
(x1, y1) = np.min(x), np.min(y)
(x2, y2) = np.max(x), np.max(y)
cropped_image = gray[x1:x2+1, y1:y2+1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.show()

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
#print(result)

text = result[0][-2]
#print(text)
img = cv2.rectangle(img, (xx, yy), (xx+w, yy+h), (0, 0, 255), 3)
img = cv2.putText(img, text, (xx, yy-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

print("License Plate: ", text)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
