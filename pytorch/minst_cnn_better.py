import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.dropout = torch.nn.Dropout2d(0.25)
        self.fc = torch.nn.Linear(288, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.fc2(x)
        output = torch.nn.x = torch.nn.functional.log_softmax(x, dim=1)
        return output

def train(model, train_loader, optimizer, epochs, log_interval):
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Clear gradient
            optimizer.zero_grad()

            # Forward propagation
            output = model(data)

            # Negative log likelihood loss
            # loss = F.nll_loss(output, target)
            loss = F.cross_entropy(output, target)

            # Back propagation
            loss.backward()

            # Parameter update
            optimizer.step()

            # Log training info
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # disable gradient calculation for efficiency
        for data, target in test_loader:
            # Prediction
            output = model(data)

            # Compute loss & accuracy
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # how many predictions in this batch are correct

    test_loss /= len(test_loader.dataset)

    # Log testing info
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Training settings
BATCH_SIZE = 64
EPOCHS = 2
LOG_INTERVAL = 10

# Define image transform
transform=transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # mean and std for the MNIST training set
    # transforms.Normalize((0.1736,), (0.3248,)) # mean and std for the EMNIST training set
])

# Load dataset
train_dataset = datasets.MNIST('./data', train=True, download=True,
                    transform=transform)
test_dataset = datasets.MNIST('./data', train=False,
                    transform=transform)
print(train_dataset.classes)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
# Create network & optimizer
model = Net()
optimizer = optim.Adam(model.parameters())

# Train
train(model, train_loader, optimizer, EPOCHS, LOG_INTERVAL)

# Save and load model
# torch.save(model.state_dict(), "mnist_cnn.pt")
# model = Net()
# model.load_state_dict(torch.load("mnist_cnn.pt"))

# Test
test(model, test_loader)

torch.save(model, 'mnist_cnn_model_better.pt')
