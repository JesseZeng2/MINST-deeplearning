# Import Libraries
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler  # For validation set
from sklearn import metrics

# Importing models
from Models.LeNet import LeNet
from Models.AlexNet import AlexNet
from Models.VGG16 import VGG16
from Models.LeNet_Improved import LeNet_Improved

# Define constants
NUM_EPOCHS = 10
BATCH_SIZE = 150
BATCH_TOTAL = 48000/BATCH_SIZE  # Train_set = 48000
BATCH_LIMIT = BATCH_TOTAL / 10
LEARNING_RATE = 0.0001

# Constants for user's input and image input
num = 0
shape1 = 32
shape2 = 32

while num == 0:
    choose = (input("\n1: LeNet\n2: AlexNet\n3: VGG-16\n4: LeNet_improved\n\nEnter 1-4 to select a model: \n"))
    if choose == "1":
        num = 1

        # Transform and reshape input
        transform = transforms.Compose(
            [transforms.Resize([shape1, shape2]),  # Resize to 32 x 32 as LeNet inputs 32x32
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
    elif choose == "2":
        num = 1
        shape1 = 227
        shape2 = 227

        # Learning rate for ALexNet to stabilize the learning curve
        LEARNING_RATE = 0.00001  # 1*10^-5

        # Transform and reshape input
        transform = transforms.Compose(
            [transforms.Resize([shape1, shape2]),  # Resize to 227 x 227 as AlexNet inputs 227x227
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
    elif choose == "3":
        num = 1
        shape1 = 224
        shape2 = 224

        # Unique constants for VGG-16 due to limited hardware
        # Learning rate must be less than 1*10^-4 for model to function properly
        LEARNING_RATE = 0.000001  # 1*10^-6
        BATCH_SIZE = 4
        BATCH_TOTAL = 48000/BATCH_SIZE
        BATCH_LIMIT = BATCH_TOTAL/10

        # Transform and reshape input
        transform = transforms.Compose(
            [transforms.Resize([shape1, shape2]),  # Resize to 224 x 224 as VGG-16 inputs 224x224
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
    elif choose == "4":
        num = 1

        # Transform and reshape input
        transform = transforms.Compose(
            [transforms.Resize([shape1, shape2]),  # Resize to 32 x 32 as LeNet inputs 32x32
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
    else:
        print("\nError: Enter a number between 1-4\n")


# Download the dataset MNIST and transform data into tensors and normalise the pixels
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Preparing for validation test
indices = list(range(len(trainset)))
np.random.shuffle(indices)
# To get 20% (12,000) of the train set (60,000) to validation set
# Using the SubsetRandomSampler function
# the number of samples per class in the validation set is proportional to the number in train set.
split = int(np.floor(0.2*len(trainset)))
train_sample = SubsetRandomSampler(indices[split:])
valid_sample = SubsetRandomSampler(indices[:split])

"""
Dataset Ratios
Train Set: 48000 (68.57%)
Validation Set: 12000 (17.14%)
Test Set: 10000 (14.29%)
"""
print("\n----Dataset Ratios----")
print("Total dataset size: " + str(len(trainset)+len(testset)))
print("Train Set: " + "{:.2f}".format((len(train_sample))/70000*100) + "%")
print("Validation Set: " + "{:.2f}".format((len(valid_sample))/70000*100) + "%")
print("Test Set: " + "{:.2f}".format((len(testset))/70000*100) + "%\n")

# Data Loader
trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=BATCH_SIZE)
validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=BATCH_SIZE)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

for batch_1 in trainloader:
    batch = batch_1
    break

print(batch[0].shape)  # as batch[0] contains the image pixels -> tensors
print(batch[1])  # batch[1] contains the labels -> tensors


# Selecting models
if choose == "1":
    net = LeNet()
    print(net)
    print("\nLeNet activated!\n")
elif choose == "2":
    net = AlexNet()
    print(net)
    print("\nAlexNet activated!\n")
elif choose == "3":
    net = VGG16()
    print(net)
    print("\nVGG16 activated!\n")
elif choose == "4":
    net = LeNet_Improved()
    print(net)
    print("\nLeNet_Improved activated!\n")


# loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# If GPU is available, then use GPU, else use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)


# function to calculate accuracy
def calc_acc(loader):
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for data in loader:
        # Sends the batch image to device and uses the model to make a good prediction
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        outputs.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Convert tensor to list and sum all values
        y_true = y_true + labels.tolist()
        y_pred = y_pred + predicted.tolist()
    # Print the confusion matrix
    print(metrics.confusion_matrix(y_true, y_pred))
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_true, y_pred, digits=3))
    return (100 * correct) / total


def train():
    train_loss = []
    valid_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        running2_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            #  Set the gradients to zero before starting to do back propragation
            #  because PyTorch accumulates the gradients on subsequent backward passes.
            optimizer.zero_grad()
            # Forward pass
            outputs = net(inputs)
            loss = loss_function(outputs, labels)

            # Back Propagation
            loss.backward()
            # Performs a parameter update based on the current gradient
            optimizer.step()

            # Updating loss
            running_loss += loss.item()

            # Print for mini batches
            if i % BATCH_LIMIT == BATCH_LIMIT - 1:  # every x mini batches
                print('[Epoch %d, %d / %d Mini Batches] loss: %.3f' %
                      (epoch + 1, i + 1, BATCH_TOTAL, running_loss / (i + 1)))

        # Calculate valid loss
        for j, data in enumerate(validloader, 0):
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            running2_loss += loss.item()

        # Calculate and output the loss and accuracy values
        train_loss.append(running_loss/len(trainloader.sampler))
        valid_loss.append(running2_loss/len(validloader.sampler))
        train_acc.append(calc_acc(trainloader))
        test_acc.append(calc_acc(testloader))

        print('Epoch: %d of %d, Train Acc: %0.3f, Test Acc: %0.3f, Loss: %0.3f'
              % (epoch + 1, NUM_EPOCHS, train_acc[epoch], test_acc[epoch], running_loss / BATCH_TOTAL))

    return train_loss, valid_loss, train_acc, test_acc

# Starting the models
start = time.time()
train_loss, valid_loss, train_acc, test_acc = train()
end = time.time()
print('%0.2f minutes' % ((end - start) / 60))

# Make Loss vs Epoch graph
plt.figure()
plt.title('Loss vs Epoch')
plt.plot(train_loss, label='Train Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Loss.png')
plt.show()

# Make accuracy vs epoch graph
plt.figure()
plt.title('Accuracy Vs Epoch')
plt.plot(train_acc, label='train accuracy')
plt.plot(test_acc, label='test accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('Accuracy.png')
plt.show()