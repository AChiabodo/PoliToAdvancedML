import torch.nn as nn
from torch.autograd import Function
# Define AlexNet architecture class
class AlexNet(nn.Module):
    def __init__(self,alpha : float, num_classes=1000):
        super(AlexNet, self).__init__()
        self.alpha = alpha
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # Category classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        # Domain classifier
        self.domain = nn.Sequential(
        nn.Linear(256 * 6 * 6, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(True),
        nn.Linear(100, 2),
        nn.LogSoftmax(dim=1)
        )
    

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        class_outputs = self.classifier(features)
        reverse_features = ReverseLayer.apply(features, self.alpha)
        domain_outputs = self.domain(reverse_features)
        return class_outputs, domain_outputs
    
    def change_last_layer(self,num_classes : int):
        self.classifier[-1] = nn.Linear(4096, num_classes)

    def get_parameters(self):
        return self.parameters()
    
    @property
    def num_classes(self):
        return self.num_classes

class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None