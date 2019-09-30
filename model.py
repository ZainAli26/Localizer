import torch
import torch.nn as nn
import torchvision

class Localizer():
    def __init__(self):
        model_ft = torchvision.models.resnet18(pretrained=True)
        ct = 0
        for child in model_ft.children():
            ct += 1
            if ct < 8:
                for param in child.parameters():
                    param.requires_grad = False
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, 64)
        fc2 = nn.Linear(64, 4)
        self.model = nn.Sequential(model_ft, fc2)
    def getModel(self):
        return self.model
        