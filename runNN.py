from cv2 import _InputArray_STD_ARRAY_MAT
from sklearn.utils import class_weight
from model import *
from data import *
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
import time
from sklearn.utils.class_weight import compute_class_weight

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

# computes class weights since our dataset is terribly imbalanced
train_labels = np.array(train_set['label'])
class_weights= compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)

print(f"Class imbalance: \n{ train_set['label'].value_counts(normalize=True) }")
print(f"Class weights: {class_weights}")
print(len(train_set), len(test_set))


train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512,512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
 
test_transform =transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


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
    batch_size=5,
    shuffle=True
)

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

devices = [print(i, torch.cuda.get_device_name(torch.cuda.device(i))) for i in range(torch.cuda.device_count())]
#device_names  = [print(i, torch.cuda.get_device_name(d)) for i, d in enumerate(devices)]


device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {device} device')
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
model = ConvNN(5, 0.5)
summary(model, input_size=(BATCH_SIZE,3,512,512))

model.apply(weights_init)


# gotta move stuff to device first
model.to(device)

loss_fn = nn.CrossEntropyLoss(class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train_loop and test_loop is from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    check = 100
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        inputs, labels = data[0].to(device), data[1].to(device)
        pred = model(inputs)
        loss = loss_fn(pred, labels)
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        train_loss += loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % check == check-1:
            loss, current = loss.item(), (batch*BATCH_SIZE) + len(inputs)
            print(f"batch loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}")
    return correct, train_loss


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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct, test_loss

epochs = 100
tt = np.arange(0, epochs)

train_loss = np.zeros((epochs, 1))
train_error = np.zeros((epochs, 1))

test_loss = np.zeros((epochs,1))
test_error = np.zeros((epochs, 1))

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    start = time.time()
    train_error[t], train_loss[t] = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_error[t], test_loss[t] = test_loop(test_dataloader, model, loss_fn)
    print(f"Epoch training time: {time.time()-start}\n")
    if epochs % 10 == 9:
        torch.save(model.state_dict(), 'model' + str(time.now()) + '.pt')        

torch.save(model.state_dict(), 'model' + str(time.now()) + '.pt')

print(f'train loss {train_loss}')
print(f'train error {train_error}')

print(f'test loss {test_loss}')
print(f'test error {test_error}')
print("Done!")
plt.figure(1)
plt.plot(tt, train_loss, 'b', tt, test_loss, 'r')
plt.figure(2)
plt.plot(tt, train_error, 'b', tt, test_error, 'r')
plt.show()
