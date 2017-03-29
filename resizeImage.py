from PIL import Image
import cv2

size = 7016, 4961
im = Image.open("test4.bin.jpg")
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save("test4.bin.resized.jpg", "JPEG", dpi=(300, 300))
print(im_resized.info)
print("done")

cv2_im = cv2.imread('test4.bin.jpg')
im_clean = cv2.fastNlMeansDenoising(cv2_im, None, 30, 31, 43)
cv2.imwrite('test4.bin.clean.jpg',im_clean)
print("done cleanning")