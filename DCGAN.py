import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,random_noise = 100) :
        super(Generator,self).__init__()
        self.fc = nn.Sequential(
            # 100 -> 8*8*1024
            nn.Linear(random_noise, 8*8*1024),
            nn.BatchNorm1d(8*8*1024),
            nn.ReLU(True)
        )
        self.gen = nn.Sequential(
            # 8*8*1024 -> 16*16*512
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 16*16*512 -> 32*32*256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 32*32*256 -> 64*64*3
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 8, 8)  # Corrected reshape
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input 64*64*3 image -> feature map 32*32*256
            nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # 32*32*256 -> 16*16*512
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # 16*16*512 -> 8*8*1024
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.disc(z)
        return self.fc(x)
