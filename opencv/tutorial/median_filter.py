import cv2
import numpy as np

salted_img = cv2.imread('opencv/imgs/Noise_salt_and_pepper.png')
salted_img = cv2.cvtColor(salted_img, cv2.COLOR_BGR2GRAY) # 將彩色通道轉換為灰階

w = salted_img.shape[1]
h = salted_img.shape[0]

filtered_img = np.zeros((h, w), np.uint8)

for x in range(1, w-1):
    for y in range(1, h-1):
        pixels = []
        for i in range(3):
            for j in range(3):
                pixels.append(salted_img[y+i-1, x+j-1])
        pixels.sort()
        filtered_img[y, x] = pixels[4]

cv2.imshow('noise', salted_img)
cv2.imshow('clean', filtered_img)

cv2.waitKey()
