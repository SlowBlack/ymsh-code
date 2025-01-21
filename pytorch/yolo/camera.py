import cv2
import numpy as np

def method1(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (640, 480))
    
    w = gray_img.shape[1]
    h = gray_img.shape[0]

    edge_img = np.zeros((h, w), np.uint8)
    mask = [-1.0, 0.0, 1.0]

    for x in range(1, w-1):
        for y in range(1, h-1):
            acc_x = 0.0
            acc_y = 0.0
            for i in range(3):
                acc_x += gray_img[y, x+i-1] * mask[i]
                acc_y += gray_img[y+i-1, x] * mask[i]
            edge_density = int((abs(acc_x) + abs(acc_y))/2)
            if edge_density > 255:
                edge_density = 255
            edge_img[y, x] = edge_density

    return edge_img

def method2(img):
    return cv2.Sobel(img, -1, 1, 1)


camera = cv2.VideoCapture(0)
while camera.isOpened:
    _, frame = camera.read()

    edge_img = method2(frame)

    cv2.imshow('camera', frame)
    cv2.imshow('edge', edge_img)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break
