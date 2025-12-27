import torch
from torchvision import models
import torch.nn as nn

sd = torch.load('model_multiclass.pth', map_location='cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(sd)
model.eval()
# dummy input
x = torch.randn(1,3,224,224)
with torch.no_grad():
    out = model(x)
print('Forward OK, logits shape:', out.shape)
