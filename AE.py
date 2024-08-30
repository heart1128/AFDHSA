import torch.nn as nn
import torch

# 定义自动编码器模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # 512x14x14 -> 256x14x14
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 256x14x14 -> 128x14x14
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 128x14x14 -> 64x14x14
            nn.ReLU(),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x14x14 -> 32x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x28x28 -> 16x56x56
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x56x156-> 3x112x112
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3x112x156-> 3x224x224
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 编码
        x = self.encoder(x)
        avgpool_x = self.avgpool(x)
        out = torch.flatten(avgpool_x, 1)
        # 解码
        x = self.decoder(x)
        return out, x # hash code & x