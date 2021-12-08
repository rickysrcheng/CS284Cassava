from cv2 import _InputArray_STD_ARRAY_MAT
from model import *
from data import *
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

img_folder = 'cassava-leaf-disease-classification\\train_images'
BATCH_SIZE = 16
'''
df = pd.read_csv('cassava-leaf-disease-classification\\train.csv')
train_set, test_set = train_test_split(df, test_size=0.2)
train_set.to_csv('train_set.csv')
test_set.to_csv('test_set.csv')
'''

train_set = pd.read_csv('train_set.csv')
test_set = pd.read_csv('test_set.csv')

print(len(train_set), len(test_set))


train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((600,800)),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomRotation(degrees=45),
                transforms.ToTensor()])
 
test_transform =transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((600,800)),
                transforms.ToTensor()])


train_dataset=ImageDataset(train_set,img_folder,train_transform)
test_dataset=ImageDataset(test_set,img_folder,test_transform)
train_dataset[50]
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
 
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=4,
    shuffle=True
)

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {device} device')
print(torch.cuda.device_count())


model = ConvNN(5, 0.5)


model.to(device)
#summary(model, input_size=(32,3,600,800))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')