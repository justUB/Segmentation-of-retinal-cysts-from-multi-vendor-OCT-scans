class reXcep(nn.Module):
  def __init__(self,inp_channels):
    super().__init__()
    self.inp_channels = inp_channels

  def forward(self,inp):
    l = nn.Conv2d(self.inp_channels, 64, kernel_size=3, stride=1, bias=False, padding='same' )(inp) 
    l = nn.batchNorm2d(64)(l)
    l = nn.SiLU()(l)

    l = nn.Conv2d(64, 128, kernel_size=3, bias=False, padding='same' )(l) 
    l = nn.batchNorm2d(128)(l)
    l = nn.SiLU()(l)

    r = nn.Conv2d(128, 256, kernel_size=1, stride=2,padding='same', bias=False)(l)
    r = nn.batchNorm2d(256)(r)

    #block2
    l = SeparableConv2d(256, 256, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(256)(l)
    l = nn.SiLU()(l)

    l = SeparableConv2d(256, 256, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(256)(l)
    skip1 = l
    l = nn.MaxPool2d(3, stride=2,padding='same')(l)

    l = l+r
    r = nn.Conv2d(256, 512, kernel_size=1, stride = 2, bias=False, padding='same' )(l)
    r = nn.batchNorm2d(512)(r)

    #Block3
    l = nn.SiLU()(l)
    l = SeparableConv2d(512, 512, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(512)(l)
    l = nn.SiLU()(l)
    l = SeparableConv2d(512, 512, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(512)(l)
    skip2 = l
    l = nn.MaxPool2d(3, stride=2,padding='same')(l)
    l = l+r

    r = nn.Conv2d(512, 1024, kernel_size=1, stride = 2, bias=False, padding='same' )(l)
    r = nn.batchNorm2d(1024)(r)

    #Block4
    l = nn.SiLU()(l)
    l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(1024)(l)
    l = nn.SiLU()(l)
    l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(1024)(l)
    skip3 = l
    l = nn.MaxPool2d(3, stride=2,padding='same')(l)
    l = l+r

    #Block5
    for i in range(1):
      r = l
      l = nn.SiLU()(l)
      l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
      l = nn.batchNorm2d(1024)(l)
      l = nn.SiLU()(l)
      l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
      l = nn.batchNorm2d(1024)(l)
      l = nn.SiLU()(l)
      l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
      l = nn.batchNorm2d(1024)(l)
      l = l+r

    #Block13
    r = nn.Conv2d(1024, 1024, kernel_size=1, stride = 2, bias=False, padding='same' )(l)
    r = nn.batchNorm2d(1024)(r)

    l = nn.SiLU()(l)
    l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(1024)(l)
    l = nn.SiLU()(l)
    l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(1024)(l)
    skip4 = l
    l = nn.MaxPool2d(3, stride=2,padding='same')(l)
    l = l+r

    l = SeparableConv2d(1024, 2048, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(2048)(l)
    l = nn.SiLU()(l)
    l = SeparableConv2d(2048, 4096, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(4096)(l)
    l = nn.SiLU()(l)

    #Rev Deconv
    l = SeparableConv2d(4096, 2048, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(2048)(l)
    l = nn.SiLU()(l)

    l = nn.Upsample(size=(2,2), mode="bilinear")(l)
    r = nn.Conv2d(1024, 1024, kernel_size=1, stride = 1, bias=False, padding='same' )(l)
    r = nn.batchNorm2d(1024)(r)

    l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(1024)(l)
    l = nn.SiLU()(l)

    l = l+skip4

    l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(1024)(l)
    l = nn.SiLU()(l)
    l = l+r

    #Rev middle
    for i in range(1):
      r = l
      l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
      l = nn.batchNorm2d(1024)(l)
      l = nn.SiLU()(l)
      l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
      l = nn.batchNorm2d(1024)(l)
      l = nn.SiLU()(l)
      l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
      l = nn.batchNorm2d(1024)(l)
      l = nn.SiLU()(l)
      l = l+r

      #Block4
    l = nn.Upsample(size=(2,2), mode="bilinear")(l)

    r = nn.Conv2d(1024, 1024, kernel_size=1, stride = 1, bias=False, padding='same' )(l)
    r = nn.batchNorm2d(1024)(r)

    l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(1024)(l)
    l = nn.SiLU()(l)

    l = l+skip3

    l = SeparableConv2d(1024, 1024, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(1024)(l)
    l = nn.SiLU()(l)
    l = l+r

    #Block3
    l = nn.Upsample(size=(2,2), mode="bilinear")(l)

    r = nn.Conv2d(1024, 512, kernel_size=1, stride = 1, bias=False, padding='same' )(l)
    r = nn.batchNorm2d(512)(r)

    l = SeparableConv2d(512, 512, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(512)(l)
    l = nn.SiLU()(l)

    l = l+skip2

    l = SeparableConv2d(512, 512, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(512)(l)
    l = nn.SiLU()(l)
    l = l+r

    #Block2
    l = nn.Upsample(size=(2,2), mode="bilinear")(l)

    r = nn.Conv2d(512, 256, kernel_size=1, stride = 1, bias=False, padding='same' )(l)
    r = nn.batchNorm2d(512)(r)

    l = SeparableConv2d(256, 256, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(512)(l)
    l = nn.SiLU()(l)

    l = l+skip1

    l = SeparableConv2d(256,256, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(256)(l)
    l = nn.SiLU()(l)
    l = l+r

    #Block1
    l = nn.SiLU()(l)
    l = SeparableConv2d(256, 128, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(128)(l)
    l = nn.SiLU()(l)
    l = SeparableConv2d(128, 64, kernel_size=3, padding='same' )(l)
    l = nn.batchNorm2d(64)(l)
    l = nn.Conv2d(64, 3, kernel_size=1, padding='same')(l)
    l = nn.SiLU()(l)
    return l



