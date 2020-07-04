import torch
from model import *
from data import *

epochs = 15
costs = []

for iterations in range(epochs):

    for image_number in range(len(train_set.index)):

        image = np.array(list(train_set[image_number])).reshape(28,28)
        image = image.float()

        gaussian_noise = torch.randn(image.shape)
        noisy_image = image + 100*gaussian_noise

        latent_space = net.encoder(noisy_image)
        reconstruction = net.decoder(latent_space)

        loss = loss_fn(image, reconstruction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        costs.append(loss.item())
    

