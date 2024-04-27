import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path (Encoder)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.AvgPool2d(2)

        # Expansive path (Decoder)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(256 + 512, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(128 + 256, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Expansive path
        up3 = self.up3(enc4)
        diffY = enc3.size()[2] - up3.size()[2]
        diffX = enc3.size()[3] - up3.size()[3]
        up3 = F.pad(up3, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.up2(dec3)
        diffY = enc2.size()[2] - up2.size()[2]
        diffX = enc2.size()[3] - up2.size()[3]
        up2 = F.pad(up2, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        diffY = enc1.size()[2] - up1.size()[2]
        diffX = enc1.size()[3] - up1.size()[3]
        up1 = F.pad(up1, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1) * x
    
if __name__ == "__main__":
    model = UNet(3, 3)
    model.to("cuda")
    x = torch.rand(3, 1063, 1600).to("cuda")
    y = model(x.unsqueeze(0)).squeeze(0)
    print(y.shape)
