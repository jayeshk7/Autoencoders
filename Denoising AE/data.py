import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

train_set = pd.read_csv('/home/kandpal/Downloads/mnist_train.csv', header=None, dtype='float64')
test_set = pd.read_csv('/home/kandpal/Downloads/mnist_test.csv', header=None, dtype='float64')

train_labels = train_set[0]
test_label = test_set[0]

train_set.drop(columns=0, inplace=True)
test_set.drop(columns=0, inplace=True)

## DataLoader part below ##