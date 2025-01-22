import numpy as np
import torch
import matplotlib.pyplot as plt

# Generate data points
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)

# Convert data to tensors
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Define a neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 50)
        self.hidden2 = torch.nn.Linear(50, 50)
        self.hidden3 = torch.nn.Linear(50, 50)
        self.output = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden1(x))
        x = torch.nn.functional.relu(self.hidden2(x))
        x = torch.nn.functional.relu(self.hidden3(x))
        x = self.output(x)
        return x


# Initialize the network and loss function
net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Train the network and visualize the results every 5 steps
for i in range(5000):
    optimizer.zero_grad()
    outputs = net(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if i % 5 == 0:
        plt.cla()
        plt.scatter(x, y, label='True Function')
        plt.plot(x, outputs.detach().numpy(), 'b', label='Prediction')
        plt.legend()
        plt.pause(0.1)
plt.show()
