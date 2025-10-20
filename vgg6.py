class VGG6(nn.Module):
    def __init__(self, activation_fn=nn.ReLU):
        super(VGG6, self).__init__()
        self.act = activation_fn()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            self.act,
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.act,
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.act,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.act,
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(8*8*256, 512),
            self.act,
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
