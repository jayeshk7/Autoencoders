import torch 
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.linear = nn.Linear(20*24*24, 16)
        self.convtrans1 = nn.ConvTranspose2d(1, 10, 1)
        self.recontruction = nn.Linear(10*4*4, 784)

    def encoder(self, image):

        image = image.unsqueeze(0)

        conv1 = F.relu( self.conv1(image) )
        conv2 = F.relu( self.conv2(conv1) )
        bottleneck = F.relu( self.linear( conv2.reshape(-1) ) )

        return bottleneck

    def decoder(self, bottleneck):

        deconv1 = F.relu( self.convtrans1( bottleneck.reshape(4, 4) ) )
        reconstr = F.relu( self.recontruction( deconv1.reshape(-1) ) )

        return reconstr


net = Model()

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters())