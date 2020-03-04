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
