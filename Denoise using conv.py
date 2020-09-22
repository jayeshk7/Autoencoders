#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import torch 
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


train_set = pd.read_csv('/home/kandpal/Downloads/mnist_train.csv', header=None, dtype='float64')
test_set = pd.read_csv('/home/kandpal/Downloads/mnist_test.csv', header=None, dtype='float64')

train_labels = train_set[0]
test_label = test_set[0]

train_set.drop(columns=0, inplace=True)
test_set.drop(columns=0, inplace=True)

## Add dataLoader part below ##


# In[3]:


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.linear1 = nn.Linear(8000, 4000)
        self.linear2 = nn.Linear(4000, 25)
        self.convtrans1 = nn.ConvTranspose2d(1, 10, 1)
        self.convtrans2 = nn.ConvTranspose2d(10, 30, 3)
        self.linear3 = nn.Linear(1470, 800)
        self.reconst = nn.Linear(800, 784)

    def encoder(self, image):

        image = torch.unsqueeze(torch.unsqueeze(image, 0), 0)

        conv1 = F.relu( self.conv1(image) )
        conv2 = F.relu( self.conv2(conv1) )
        linear = F.relu( self.linear1( conv2.reshape(-1) ) )
        bottleneck = F.relu( self.linear2( linear ) )

        return bottleneck

    def decoder(self, bottleneck):
        
        bottleneck = bottleneck.reshape(5, 5)
        bottleneck = torch.unsqueeze(torch.unsqueeze(bottleneck, 0), 0)

        deconv1 = F.relu( self.convtrans1( bottleneck ) )
        deconv2 = F.relu( self.convtrans2( deconv1 ))
        reconstr = F.relu( self.linear3( deconv2.reshape(-1) ) )
        reconstr = F.relu( self.reconst( reconstr ) )

        return reconstr


# In[4]:


net = Model().cuda()

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters())
epochs = 10


# In[5]:


costs = []

for iterations in range(epochs):

    for image_number in range(len(train_set.index)):

        image = torch.from_numpy(np.array(list(train_set.iloc[1])).reshape(28,28))
        image = image.float()

        gaussian_noise = torch.randn(image.shape)
        noisy_image = image + 100*gaussian_noise

        latent_space = net.encoder(noisy_image.cuda())
        reconstruction = net.decoder(latent_space.cuda())

        loss = loss_fn(image.flatten().cuda(), reconstruction.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        costs.append(loss.item())
    print(f'average loss {np.mean(costs)}')
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(image.cpu().detach().reshape(28,28).numpy())

    plt.subplot(132)
    plt.imshow(noisy_image.reshape(28,28))

    plt.subplot(133)
    plt.imshow(reconstruction.cpu().detach().reshape(28,28).numpy())


# In[ ]:


from scipy.ndimage import gaussian_filter
plt.plot(costs)
plt.plot(gaussian_filter(costs, 10))
plt.show()

