import cv2
import numpy as np

# Constants
w = 512
h = 512
rect_size = 20
rect_orig_x = w // 2 - rect_size // 2
rect_orig_y = h // 2 - rect_size // 2

# Variables
offset_x = 0
offset_y = 0
direction = 0 # 0:預設值，不動; 1:上; 2:下; 3:左; 4:右

while True:
    img = np.zeros((h, w), np.uint8)

    # cv2.rectangle(img, point1, point2, color, border_thickness)
    cv2.rectangle(img, (rect_orig_x+offset_x, rect_orig_y+offset_y), (rect_orig_x+rect_size+offset_x, rect_orig_y+rect_size+offset_y), 255, -1)

    # 依照direction更新offset
    if direction == 1:
        offset_y -= 1
    elif direction == 2:
        offset_y += 1
    elif direction == 3:
        offset_x -= 1
    elif direction == 4:
        offset_x += 1

    cv2.imshow('rect', img)
    key = cv2.waitKey(33)

    if key == ord('q'): # Leave the Loop
        break
    elif key == ord('w'):
        direction = 1
    elif key == ord('s'):
        direction = 2
    elif key == ord('a'):
        direction = 3
    elif key == ord('d'):
        direction = 4
