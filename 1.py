import cv2
import time
import argparse
import imutils

net = cv2.dnn.readNetFromTorch(r'C:\Users\dhabr\OneDrive\Desktop\nst_opencv\triangle_style_2000.t7')
image = cv2.imread(r'C:\Users\dhabr\OneDrive\Desktop\nst_opencv\content_2.jpeg')
image = imutils.resize(image, width=600)

(h, w) = image.shape[:2]
print(h,w)

blob = cv2.dnn.blobFromImage(image, 1.0, (w,h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
net.setInput(blob)

start = time.time()
output = net.forward()
end = time.time()

output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.939
output[1] += 116.779
output[2] += 123.680

output /= 255.0
output = output.transpose(1,2,0)

print(f'nst took {end-start} seconds')

cv2.imshow('input', image)
cv2.imshow('output', output)
cv2.waitKey(0)



