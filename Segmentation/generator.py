"""
Team:
KANTAMNENI MITHRA SAI TEJA - S20180010074
MANEPALLI UDAY BHASKAR - S20180010099
VASIREDDY KOMAL KUMAR - S20180010189
PAIDALA VIKRANTH REDDY - S20180010126
"""

import torch
import torch.nn as nn

class gen(nn.Module):
    def __init__(self,inp_channels):
        super().__init__()
        self.d1 = nn.Sequential(nn.Conv2d(inp_channels, 64, kernel_size=4, stride=2, padding=1 ), nn.LeakyReLU(0.2))

        self.d2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 4, stride =2, padding =1), nn.LeakyReLU(0.2))
        self.d3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 4, stride =2, padding =1), nn.LeakyReLU(0.2))
        self.d4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = 4, stride =2, padding =1), nn.LeakyReLU(0.2))
        self.d5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 4, stride =2, padding =1), nn.LeakyReLU(0.2))
        self.d6 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 4, stride =2, padding =1), nn.LeakyReLU(0.2))
        self.d7 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 4, stride =2, padding =1), nn.LeakyReLU(0.2))

            #bottleneck
        self.b = nn.Sequential(nn.Conv2d(512,512, kernel_size = 4, stride =2, padding =1), nn.ReLU())

            #upsampling
        self.u1 = nn.Sequential(nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1, bias=False), nn.BatchNorm2d(512),nn.ReLU(True), nn.Dropout(0.5))
        self.u2 = nn.Sequential(nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1, bias=False), nn.BatchNorm2d(512),nn.ReLU(True), nn.Dropout(0.5))
        self.u3 = nn.Sequential(nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1, bias=False), nn.BatchNorm2d(512),nn.ReLU(True), nn.Dropout(0.5))
        self.u4 = nn.Sequential(nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.u5 = nn.Sequential(nn.ConvTranspose2d(1024,256,kernel_size=4,stride=2,padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.u6 = nn.Sequential(nn.ConvTranspose2d(512,128,kernel_size=4,stride=2,padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.u7 = nn.Sequential(nn.ConvTranspose2d(256,64,kernel_size=4,stride=2,padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.u8 = nn.Sequential(nn.ConvTranspose2d(128,3,kernel_size=4,stride=2,padding=1, bias=False), nn.Tanh())
        
    def forward(self, X):
        d1 = self.d1(X)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        b = self.b(d7)

        u1 = self.u1(b)
        u2 = self.u2(torch.cat([u1,d7], 1))
        u3 = self.u3(torch.cat([u2,d6], 1))
        u4 = self.u4(torch.cat([u3,d5], 1))
        u5 = self.u5(torch.cat([u4,d4], 1))
        u6 = self.u6(torch.cat([u5,d3], 1))
        u7 = self.u7(torch.cat([u6,d2], 1))
        u8 = self.u8(torch.cat([u7,d1], 1))

        return u8