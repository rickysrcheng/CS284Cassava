from model import *
from data import *

img_folder = 'cassava-leaf-disease-classification\\train_images'
BATCH_SIZE = 32
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
                transforms.Resize((400, 400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor()])
 
test_transform =transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.ToTensor()])


train_dataset=ImageDataset(train_set,img_folder,train_transform)
test_dataset=ImageDataset(test_set,img_folder,test_transform)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
print(torch.cuda.device_count())
