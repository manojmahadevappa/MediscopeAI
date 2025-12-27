import torch
from torchvision import models
import torch.nn as nn

m = models.resnet18(pretrained=False)
# set 3-class head
m.fc = nn.Linear(m.fc.in_features, 3)
# save state_dict for the demo multiclass model
torch.save(m.state_dict(), 'model_multiclass.pth')
print('Saved model_multiclass.pth')
