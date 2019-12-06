#%%
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
from intrprt.data.CIFAR10 import *

# If you are loading a saved trained model, set `loading` to `True`,
# and provide the correct file name and path for model name
loading = False
model_path = 'intrprt/model/'
# model_name = f'cnn_{time.strftime("%Y%m%d-%H%M%S")}.pt'
model_name = 'cnn_20191205.pt'

#%%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#%% [markdown]
## Train the CNN
# Create the network
net = Net()


#%% [markdown]
# Load the dataset.
X, y, X_test, y_test = load_CIFAR10(transform=True)

#%% [markdown]
# Transform to normalized-range tensors [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%% [markdown]
# Display some sample images
# num_samples = 3
# indices = np.random.randint(0, X.shape[0], num_samples)
# images = list()
# labels = list()
# for i in range(num_samples):
#     images.append(transform(Image.fromarray(X[i])))
#     labels.append(classes[y[i]])

# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % labels[j] for j in range(num_samples)))

#%% [markdown]
# Create the network, loss, and optimizer.
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%% [markdown]
# Train the convolutional neural network.
# batch_size = 1
epochs = 1
# inputs = np.zeros((batch_size,) + X.shape[1:])
for epoch in range(epochs):  # loop over the dataset multiple times
    
    running_loss = 0.0
    # for i in range(0, X.shape[0], batch_size):
    # for i in range(X.shape[0]):
    for i in range(10):
        inputs = torch.tensor(np.expand_dims(transform(Image.fromarray(X[i])), axis=0))
        label = torch.from_numpy(np.array([y[i]]).astype(np.int64))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save the trained model once done trianing
torch.save(net.state_dict(), model_path + model_name)

#%% [markdown]
# Load a saved trained model
net = Net()
net.load_state_dict(torch.load(model_path + model_name))
print(net)

#%% [markdown]
# Check the prediction against some sample images from the test data
# print images
num_samples = 5
indices = np.random.randint(0, X_test.shape[0], num_samples)
images = np.zeros((num_samples,) + transform(Image.fromarray(X_test[i])).shape)
labels = list()

for i in range(num_samples):
    # images.append(transform(Image.fromarray(X_test[i])))
    images[i,:] = transform(Image.fromarray(X_test[i]))
    labels.append(classes[y_test[i]])

images = torch.from_numpy(images)
imshow(torchvision.utils.make_grid(images))
print('Ground truth:')
print(' '.join('%5s' % labels[j] for j in range(num_samples)))

outputs = net(images.float())
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(num_samples)))

#%% [markdown]
# Calculate the accuracy on test data

# %%
