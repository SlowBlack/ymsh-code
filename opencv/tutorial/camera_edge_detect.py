import cv2
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Images
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# # Inference
# results = model(imgs)

# # Results
# results.print()
# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)

camera = cv2.VideoCapture(0)
while camera.isOpened:
    _, frame = camera.read()

    results = model([frame])
    for index, result in results.pandas().xyxy[0].iterrows():
        xmin = int(result['xmin'])
        xmax = int(result['xmax'])
        ymin = int(result['ymin'])
        ymax = int(result['ymax'])
        cv2.putText(frame, result['name'], (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, '{:.2f}'.format(result['confidence']), (xmin, ymin+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('camera', frame)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break
