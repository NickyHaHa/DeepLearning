# Tune on layer and model
# Use ToTensor() easy transform the data
# Shuffle on train data in Dataloader
# Best test accuracy is 70.53%, lr = 0.015
import torch, os
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchsummary import summary

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 3 * 3 , 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 16x30x30->16x15x15
        x = self.pool(self.relu(self.conv2(x))) # 64x14x14->64x7x7
        x = self.pool(self.relu(self.conv3(x))) # 128x6x6->128x3x3
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Use own dataset
class CIFAR10Data(Dataset):
  def __init__(self, csv_file, path, transform=None):
    self.data = pd.read_csv(csv_file)
    self.path = path
    self.transform = transform
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    img_path = os.path.join(self.path, self.data.iloc[index, 0])
    image = Image.open(img_path).convert('RGB')
    label = self.data.iloc[index, 1]
    if self.transform:
      image = self.transform(image)
    return image, label

def Draw(x, trainY, valY, title):
    plt.plot(x, trainY, '-', color='#EA0000', label="T r a i n "+title)
    plt.plot(x, valY, '-', color='#0080FF', label="V a l i d a t e "+title)
    plt.xlabel("E p o c h")
    plt.ylabel(title)
    plt.legend()
    plt.savefig(title+".png")

def Train(num_epochs, batch_size, learning_rate):
    # Device check
    if torch.cuda.is_available():
      device = torch.device('cuda')
      print('Ya, You Really Have GPU!')

    train_csv = 'data/train_images/train.csv'
    train_path = 'data/train_images/'
    # CIFAR10_train_data = datasets.CIFAR10('./data', train=True, download=True, transform=ToTensor())
    CIFAR10_train_data = CIFAR10Data(train_csv, train_path, ToTensor())

    # Data split
    train_data_size = int(len(CIFAR10_train_data) * 0.9)
    validate_data_size = len(CIFAR10_train_data) - train_data_size
    CIFAR10_train_data, CIFAR10_validate_data = torch.utils.data.random_split(CIFAR10_train_data, [train_data_size, validate_data_size])

    train_loader = DataLoader(dataset=CIFAR10_train_data, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=CIFAR10_validate_data, batch_size=batch_size)

    model = ConvolutionalNeuralNetwork().to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Print the model
    summary(model.cuda(), (3, 32, 32))

    accuracy_train, accuracy_val = np.array([]), np.array([])
    loss_train, loss_val = np.array([]), np.array([])

    for epoch in range(num_epochs):

        train_correct, val_correct = 0, 0
        train_err, val_err = 0, 0

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Gradient reset
            optimizer.zero_grad()

            # FeedForward
            predicted = model(images)

            # Backforward
            loss = criterion(predicted, labels)
            loss.backward()
            train_err += loss.item()

            # Update
            optimizer.step()

            # Check correct
            predicted = torch.max(predicted, 1)[1]
            train_correct += (predicted == labels).sum().item()

        model.eval()
        for j, (images, labels) in enumerate(validate_loader):
            images, labels = images.to(device), labels.to(device)

            predicted = model(images)
            loss = criterion(predicted, labels)
            val_err += loss.item()

            # Check correct
            predicted = torch.max(predicted, 1)[1]
            val_correct += (predicted == labels).sum().item()


        train_acc = train_correct / train_data_size
        train_err = train_err / len(train_loader)
        accuracy_train = np.append(accuracy_train, train_acc)
        loss_train = np.append(loss_train, train_err)
        
        val_acc = val_correct / validate_data_size
        val_err = val_err / len(validate_loader)
        accuracy_val = np.append(accuracy_val, val_acc)
        loss_val = np.append(loss_val, val_err)
            
        # print(f'epoch {epoch+1}/{num_epochs}, train loss = {train_loss.item():.4f}, validate loss = {val_loss.item():.4f}')
        print(f'epoch {epoch+1}/{num_epochs}, train acc {100.0*train_acc:.2f}%, validate acc {100.0*val_acc:.2f}%, train loss {train_err:.4f}, validate loss {val_err:.4f}')

    print('Finished Training')

    Draw(np.linspace(0, num_epochs, num_epochs), accuracy_train, accuracy_val, "- A c c")
    plt.clf()
    Draw(np.linspace(0, num_epochs, num_epochs), loss_train, loss_val, "- L o s s")

    return model, device

def Test(model, device, batch_size):

    test_csv = 'data/test_images/test.csv'
    test_path = 'data/test_images/'
    # CIFAR10_test_data = datasets.CIFAR10('./data', train=False, download=True, transform=ToTensor())
    CIFAR10_test_data = CIFAR10Data(test_csv, test_path, ToTensor())
    test_loader = DataLoader(dataset=CIFAR10_test_data, batch_size=batch_size)

    model = model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            # Prevent data % batch_size != 0
            if len(labels) < batch_size:
                batch_size = len(labels)
            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Test accuracy of the network: {acc:.2f} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {i}: {acc} %')

if __name__ == '__main__':
    num_epochs, batch_size, learning_rate = 20, 20, 0.015
    print('Epoch', num_epochs)
    print('Batch', batch_size)
    print('Learning rate', learning_rate)
    model, device = Train(num_epochs, batch_size, learning_rate)
    Test(model, device, batch_size)
    path = '/lab4.pt'
    torch.save(model.state_dict(), path)