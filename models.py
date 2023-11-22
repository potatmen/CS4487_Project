import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision

class Models(nn.Module):
    
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.5):
        super(Models, self).__init__()
        self.modelchoice = modelchoice
        
        if modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'vgg19_bn':
            weights = VGG19_BN_Weights.DEFAULT
            self.model = torchvision.models.vgg19_bn(weights = weights)
        else:
            raise Exception('Choose valid model, e.g. resnet50')



    def forward(self, x):
        x = self.model(x)
        return x

