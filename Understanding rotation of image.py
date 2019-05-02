import numpy as np
import cv2

img1 = cv2.imread('images/e_19.jpeg')
##img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('images/5_2.jpeg')
##gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
##gray = cv2.GaussianBlur(gray, (3,3), 0)
##edged = cv2.Canny(gray, 20, 100)

cv2.imshow('edged',img2)

##cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
##                        cv2.CHAIN_APPROX_SIMPLE)
##cnts = cnts[1]
##
##if len(cnts) > 0:
##    # grab the largest contour, then draw a mask for the pill
##    c = max(cnts, key=cv2.contourArea)
##    mask = np.zeros(gray.shape, dtype="uint8")
##    cv2.drawContours(mask, [c], -1, 255, -1)
## 
##    # compute its bounding box of pill, then extract the ROI,
##    # and apply the mask
##    (x, y, w, h) = cv2.boundingRect(c)
##    imageROI = img2[y:y + h, x:x + w]
##    maskROI = mask[y:y + h, x:x + w]
##    imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)

##cv2.imshow('ffk', imageROI)

r1, c1, _ = img2.shape

##M = cv2.getRotationMatrix2D((r1/2, c1/2), 90, 1)
##img = cv2.warpAffine(imageROI, M, (r1, c1))
angles = [-30,60, -60]

count = 1
for angle in angles:
    print("angle is: ", angle)
    M = cv2.getRotationMatrix2D((r1/2, c1/2), angle, 1)
    img = cv2.warpAffine(img2, M, (r1, c1))
    cv2.imwrite('rotate_{}.jpeg'.format(count), img)
    cv2.imshow('img', img)
    count+=1
    cv2.waitKey(0)

##count = 1
##for angle in np.arange(0, 360, 30):
##    print("angle is: ", angle)
##    M = cv2.getRotationMatrix2D((r1/2, c1/2), angle, 1)
##    img = cv2.warpAffine(imageROI, M, (r1, c1))
##    cv2.imwrite('rotate2_{}.jpeg'.format(count), img)
##    cv2.imshow('img', img)
##    count+=1
##    cv2.waitKey(0)
####	
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

##
##count = 1
##for angle in np.arange(0, 360, 30):
##    print("angle is: ", angle)
##    img = rotate_bound(imageROI, angle)
##    cv2.imwrite('rotate2_{}.jpeg'.format(count), img)
##    cv2.imshow('img', img)
##    count+=1
##    cv2.waitKey(0)
