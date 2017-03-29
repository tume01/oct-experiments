import cv2

cv2_im = cv2.imread('test4.bin.jpg')
im_clean = cv2.fastNlMeansDenoising(cv2_im, None, 40, 31, 43)
cv2.imwrite('test4.bin.clean.jpg',im_clean)
print("done cleanning")