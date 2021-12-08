import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

img_folder = 'cassava-leaf-disease-classification\\train_images'

df = pd.read_csv('cassava-leaf-disease-classification\\train.csv', index_col=False)

df = df['label'].value_counts(normalize=True)
print(df)
train_set, test_set = train_test_split(df, test_size=0.2)

print(train_set['label'].value_counts(normalize=True))
print(test_set['label'].value_counts(normalize=True))

train_set.to_csv('train_set.csv', index=None)
test_set.to_csv('test_set.csv', index=None)

