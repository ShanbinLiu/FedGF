from torch import nn
import torchvision.models as models

def Model():
    resnet18 = models.resnet18()
    num_features = resnet18.fc.in_features

    # 替换最后一层为新的全连接层，输出大小为10
    resnet18.fc = nn.Linear(num_features, 10)
    model = resnet18
    return model


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)