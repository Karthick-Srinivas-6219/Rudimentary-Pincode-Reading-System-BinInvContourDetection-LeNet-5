from torch.nn import Module # base class that contains full support for NN components  - layers, params, fwd/bwd pass etc.
from torch import nn # NN toolkit for loss fns optimizers etc.


class LeNet5(Module):  # Builds the model framework - by inheriting functions etc. from Module class
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # conv1
        self.conv1 = nn.Conv2d(1, 6, 5) # (in_channel, filters, kernel_shape)
        self.relu1 = nn.ReLU()
        # maxpool1
        self.pool1 = nn.MaxPool2d(2) # (kernel_shape)

        # conv2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        # maxpool2
        self.pool2 = nn.MaxPool2d(2)

        # conv3
        self.conv3 = nn.Conv2d(16, 120, 4)
        self.relu3 = nn.ReLU() # o/p is (120, 1, 1)

        # FCL
        self.fc1 = nn.Linear(120, 84) # passes the 120-d vector into fcl1 with 84 nodes
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    # fwd prop
    def forward(self, y): # defines 'data flow' through network defined in 'LeNet5' - builds the 'data flow graph'

        y = self.pool1(self.relu1(self.conv1(y))) # conv1 + maxpool1
        y = self.pool2(self.relu2(self.conv2(y))) # conv2 + maxpool2
        y = self.relu3(self.conv3(y)) # conv3

        # flatten for FCL
        y = y.view(y.size(0), -1) # flatten to shape [batch, 120]

        y = self.relu4(self.fc1(y)) # fc1 + ReLU
        y = self.fc2(y) # fc2 raw logits computed

        return y # raw logits from fc2 are returned