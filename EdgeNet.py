import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeNet(nn.Module):
    def __init__(self, c1, c2, ks=3):  # c1是输入通道数，c2是输出通道数
        super(EdgeNet, self).__init__()
        self.conv0 = nn.Conv2d(c1, c1 // 2, kernel_size=ks, stride=1, padding=ks // 2)
        self.conv1 = nn.Conv2d(c1 // 2, c1 // 2, kernel_size=ks, stride=1, padding=ks // 2)

        self.down1 = nn.Conv2d(c1 // 2, c2, kernel_size=ks, stride=2, padding=ks // 2)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=ks, stride=2, padding=ks // 2)

        self.up1 = nn.ConvTranspose2d(c2, c2, kernel_size=ks, stride=2, padding=ks // 2, output_padding=1)  # 保持通道数不变
        self.up2 = nn.ConvTranspose2d(c1 // 2, c1 // 2, kernel_size=ks, stride=2, padding=ks // 2,
                                      output_padding=1)  # 修改这里

        self.aggD2 = nn.Conv2d(c2 * 2, c1 // 2, kernel_size=ks, stride=1, padding=ks // 2)
        self.aggD1 = nn.Conv2d(c1 // 2 * 2, c1 // 2, kernel_size=ks, stride=1, padding=ks // 2)  # 修改这里

        self.conv2 = nn.Conv2d(c2, c2, kernel_size=ks, stride=1, padding=ks // 2)

        self.img_prd = nn.Sequential(
            nn.Conv2d(c1 // 2, c1 // 2, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(c1 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c1 // 2, 3, kernel_size=ks, stride=1, padding=ks // 2)
        )

    def forward(self, x):
        E1 = self.conv0(x)
        E2 = self.down1(E1)
        E3 = self.down2(E2)

        D3 = self.conv2(E3)
        D3 = self.up1(D3)  # 上采样后直接与E2拼接
        D2 = self.aggD2(torch.cat([D3, E2], 1))  # 确保E2的尺寸与D3匹配
        D1 = self.aggD1(torch.cat([self.up2(D2), E1], 1))

        img_prd = self.img_prd(D1)
        return D1, img_prd


if __name__ == '__main__':
    input = torch.randn(8, 256, 640, 640)
    model = EdgeNet(256, 256)
    output = model(input)
    print(output[0].shape)  # 打印第一个张量的形状
    print(output[1].shape)  # 打印第二个张量的形状
