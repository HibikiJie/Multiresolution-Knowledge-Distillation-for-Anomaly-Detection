import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19


# print(vgg19())
# exit()

class SourceNetwork(nn.Module):

    def __init__(self):
        super(SourceNetwork, self).__init__()
        vgg = vgg19(True)
        layers = vgg.features
        # print(layers)
        self.layer1 = layers[:5]
        self.layer2 = layers[5:10]
        self.layer3 = layers[10:19]
        self.layer4 = layers[19:28]
        self.layer5 = layers[28:]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out2, out3, out4, out5


class ClonerNetwork(nn.Module):

    def __init__(self):
        super(ClonerNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 128, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 256, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 512, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 512, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out2, out3, out4, out5


if __name__ == '__main__':
    m = SourceNetwork()
    # print(m)
    # exit()
    x = torch.randn(1, 3, 224, 224)
    ys = m(x)
    for y in ys:
        print(y.shape)
    m = ClonerNetwork()
    # print(m)
    x = torch.randn(1, 3, 224, 224)
    ys = m(x)
    for y in ys:
        print(y.shape)
