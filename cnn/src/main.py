import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3) # filter=3x3 (shorthand 3), 6 output channels
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # input (excluding batch x 4): 3x32x32, output: pool(6x30x30)=6x15x15
        x = self.pool(F.relu(self.conv2(x))) # input: 6x15x15, output: pool(16x13x13)=16x6x6
        x = F.relu(self.conv3(x)) # input: 16x6x6, output: 32x4x4
        x = torch.flatten(x, 1) # flatten all dimensions except batch, input=32x4x4, output=512x1 
        x = F.relu(self.fc1(x)) # input: 512x1, output: 256x1
        x = F.relu(self.fc2(x)) # input: 256x1, output: 128x1
        x = self.fc3(x) # input: 128x1, output: 10x1
        return x

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 4
    dataset_folder = '../in/cifar10/'
    trainset = torchvision.datasets.CIFAR10(root=dataset_folder, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=dataset_folder, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def train(train_loader, trained_model_path):
    # define model
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # training
    total_epochs = 4
    for epoch in range(total_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data # get batch
            optimizer.zero_grad() # zero the parameter gradients
            outputs = net(inputs) # feed forward
            loss = criterion(outputs, labels) # compute loss
            loss.backward() # compute gradient
            optimizer.step() # update weights
            running_loss += loss.item() 
            if i % 2000 == 1999: # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    torch.save(net.state_dict(), trained_model_path)

def test(test_loader, trained_model_path):
    net = Net()
    net.load_state_dict(torch.load(trained_model_path))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def main():

    """
        tutorial -> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    print('cnn: starting...')

    trained_model_path = '../out/cifar_net.pth'
    train_loader, test_loader = get_data()
    train(train_loader, trained_model_path)
    test(test_loader, trained_model_path)

    print('cnn: ending...')

if __name__ == '__main__':
    main()