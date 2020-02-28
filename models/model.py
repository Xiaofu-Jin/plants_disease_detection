import torchvision
import torch
import torch.nn.functional as F 
from torch import nn
from config import config

#def generate_model():
#    class DenseModel(nn.Module):
#        def __init__(self, pretrained_model):
#            super(DenseModel, self).__init__()
#            self.classifier = nn.Linear(pretrained_model.classifier.in_features, config.num_classes)
#
#            for m in self.modules():
#                if isinstance(m, nn.Conv2d):
#                    nn.init.kaiming_normal(m.weight)
#                elif isinstance(m, nn.BatchNorm2d):
#                    m.weight.data.fill_(1)
#                    m.bias.data.zero_()
#                elif isinstance(m, nn.Linear):
#                    m.bias.data.zero_()
#
#            self.features = pretrained_model.features
#            self.layer1 = pretrained_model.features._modules['denseblock1']
#            self.layer2 = pretrained_model.features._modules['denseblock2']
#            self.layer3 = pretrained_model.features._modules['denseblock3']
#            self.layer4 = pretrained_model.features._modules['denseblock4']
#
#        def forward(self, x):
#            features = self.features(x)
#            out = F.relu(features, inplace=True)
#            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
#            out = F.sigmoid(self.classifier(out))
#            return out
#
#    return DenseModel(torchvision.models.densenet169(pretrained=True))
#
class BilinearCNN(nn.Module):
    def __init__(self):
        # Convolution and pooling layers of VGG-16.
        nn.Module.__init__(self)
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, config.num_classes)

        # Freeze all previous layers.
        #for param in self.features.parameters():
        #    param.requires_grad = False
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant(self.fc.bias.data, val=0)

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 28**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, config.num_classes)
        return X

class MyVGG(nn.Module):
    def __init__(self):
        # Convolution and pooling layers of VGG-16.
        nn.Module.__init__(self)
        self.features = torchvision.models.vgg16(pretrained=True).features

        # Linear classifier.
        self.fc = torch.nn.Linear(512*14*14, config.num_classes)

        # Freeze all previous layers.
        for param in self.features.parameters():
            param.requires_grad = False
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant(self.fc.bias.data, val=0)

    def forward(self, x):
        """Forward pass of the network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_net():
    #return MyModel(torchvision.models.resnet101(pretrained = True))
    ##model = torchvision.models.resnet50(pretrained = True)    
    #for param in model.parameters():
    #    param.requires_grad = False
    ##model.avgpool = nn.AdaptiveAvgPool2d(1)
    ##model.fc = nn.Linear(2048,config.num_classes)
    #model = BilinearCNN()
    #model = torchvision.models.vgg16(pretrained=True).features
    #model = torch.nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())[:-1])  # Remove pool5.
    model = MyVGG()
    return model

