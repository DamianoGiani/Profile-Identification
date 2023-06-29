
from torch import nn
from torchvision.models import ResNet18_Weights,resnet18
import torch.nn.functional as F

class ResProj(nn.Module):
    def __init__(self):
        super(ResProj, self).__init__()        
        self.res = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.lin1=nn.Linear(1000, 512)
        self.lin2=nn.Linear(512, 256)
        self.relu=nn.ReLU()
        self.proj=nn.Sequential(
            nn.Linear(256, 20)                     
        )
    def forward(self, x):
        
        logits = self.res(x)
        output=self.lin1(logits)
        output=self.relu(output)
        output=self.lin2(output)
        output=F.normalize(output, dim=1)
        #output=self.proj(output) #per crossentropy

        return output