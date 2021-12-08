from cv2 import _InputArray_STD_ARRAY_MAT
from model import *
from data import *
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

img_folder = 'cassava-leaf-disease-classification\\train_images'
BATCH_SIZE = 20
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
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
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
summary(model, input_size=(BATCH_SIZE,3,600,800))

model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train_loop and test_loop is from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    check = 20
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        inputs, labels = data[0].to(device), data[1].to(device)
        pred = model(inputs)
        loss = loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % check == check-1:
            loss, current = loss.item(), (batch*BATCH_SIZE) + len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data in dataloader:
            X,y = data[0].to(device), data[1].to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
