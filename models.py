import torch.nn as nn


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock1d, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=[1],
                      padding=(padding,),dilation=[1],groups=in_channels, bias=False),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(out_channels,eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False)
        )

    def forward(self, x):
        return self.layers(x)


class JasperBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(JasperBlock1d, self).__init__()

        self.blocks = nn.Sequential(
            ConvBlock1d(in_channels, out_channels, kernel_size, padding),
            ConvBlock1d(out_channels, out_channels, kernel_size, padding),
            ConvBlock1d(out_channels, out_channels, kernel_size, padding),
            ConvBlock1d(out_channels, out_channels, kernel_size, padding)
        )

        self.last = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size= kernel_size, stride=[1], padding=(padding,),
                      dilation=[1], groups=out_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(out_channels,eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        ])

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), stride=[1], bias=False),
            nn.BatchNorm1d(out_channels,eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):

        y = self.residual(x)
        x = self.blocks(x)

        for idx, layer in enumerate(self.last):
            x = layer(x)
            if idx == 2:
                x += y
                x = self.relu(x)
                x = self.dropout(x)

        return x


class QuartzNet(nn.Module):
    def __init__(self, repeat, in_channels, out_channels):
        super(QuartzNet, self).__init__()

        block_channels = [256, 256, 512, 512, 512]
        block_k = [33, 39, 51, 63, 75]

        self.C1 = nn.Sequential(
            nn.Conv1d(in_channels, block_channels[0], kernel_size=33, padding=16, bias=False),
            nn.BatchNorm1d(block_channels[0], eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.3, inplace=False)
        )

        self.B = nn.ModuleList([])

        for i in range(5):
            num_in = block_channels[i]
            num_out = block_channels[i+1]
            pad = block_k[i] // 2
            k = block_k[i]

            self.B.append(JasperBlock1d(num_in, num_out, k, pad))

            for rep in range(repeat):
                self.B.append(JasperBlock1d(num_out, num_out, k, pad))

        self.C2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=87, padding=86, dilation=2),
            nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False)
        )

        self.C3 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False)
        )

        self.C4 = nn.Conv1d(1024, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.C1(x)

        for block in self.B:
            x = block(x)

        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)

        return x