import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2d(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)
                 ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.maxpool2d = nn.MaxPool2d(stride=2, kernel_size=2)


        # Down part of UNET
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f

        # Up part of UNET
        for f in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    f * 2, f, stride=2, kernel_size=2
                )
            )
            self.ups.append(DoubleConv(f * 2, f))

        # Bottleneck part of the UNET
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Final output part of the UNet

        self.final_conv2d = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connection = []

        for down in self.downs:
            x = down(x)
            skip_connection.append(x)
            x = self.maxpool2d(x)

        x = self.bottleneck(x)
        skip_connection = skip_connection[::-1]


        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            sc = skip_connection[idx // 2]
            # Before concatinating let's make sure the dimensions are same
            if x.shape != sc.shape:
                F.resize(x,size=sc.shape[2:] )

            concat_skip = torch.cat((sc, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv2d(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    pred = model(x)
    print(pred.shape)
    print(x.shape)
    assert pred.shape == x.shape


if __name__ == "__main__":
    ## working tested on april 11, 14:50
    test()