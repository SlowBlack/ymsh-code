import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.main = torch.nn.Sequential(
#             torch.nn.Linear(in_features=784, out_features=128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(in_features=128, out_features=64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(in_features=64, out_features=10),
#             torch.nn.LogSoftmax(dim=1)
#         )

#     def forward(self, input):
#         return self.main(input)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 32, 3)
        self.dropout = torch.nn.Dropout2d(0.25)
        self.fc = torch.nn.Linear(800, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = torch.nn.x = torch.nn.functional.log_softmax(x, dim=1)
        return output
model = torch.load('mnist_cnn_model.pt')

def create_canvas():
  return np.zeros((512, 512), dtype=np.uint8)

current_frame = create_canvas()

press_flag = False
def draw(event, x, y, flags, param):
  global press_flag
  if event == cv2.EVENT_LBUTTONDOWN:
    press_flag = True
  elif event == cv2.EVENT_LBUTTONUP:
    press_flag = False
    feed_img = cv2.resize(current_frame, (28, 28))
    feed_tensor = transforms.ToTensor()(feed_img)
    # CNN
    feed_tensor = torch.unsqueeze(feed_tensor, dim=1)

    # DNN
    # feed_tensor = torch.flatten(feed_tensor)
    # feed_tensor = torch.unsqueeze(feed_tensor, dim=0)

    predict = model(feed_tensor)
    print(predict)
    print('Ans: {}'.format(torch.argmax(predict)))
  elif event == cv2.EVENT_MOUSEMOVE and press_flag:
    cv2.circle(current_frame, (x,y), 10, 255, -1)

def clear_canvas():
  global current_frame
  current_frame = create_canvas()

cv2.namedWindow('draw')
cv2.setMouseCallback('draw', draw)

while True:
  cv2.imshow('draw', current_frame)
  key = cv2.waitKey(33)
  if key == ord('c'):
    clear_canvas()
  elif key == ord('q'):
    break
