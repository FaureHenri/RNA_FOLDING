import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fc_block, self).__init__()
        self.dropout = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(ch_in, ch_out)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class U_Net(nn.Module):
    def __init__(self, img_ch=[17, 32, 64, 128], output_ch=3):
        super(U_Net, self).__init__()
        kernel = 2
        stride = 2
        # output_size = (input - kernel_size + 2 * padding) / stride + 1
        block_wd = 148
        self.layers = nn.Sequential()
        for th, ch in enumerate(img_ch):
            if th < len(img_ch) - 2:
                self.layers.add_module(name='Conv' + str(th + 1), module=conv_block(ch_in=ch, ch_out=img_ch[th + 1]))
                block_wd_test = (block_wd - kernel) / 2 + 1
                if isinstance(block_wd_test, int):
                    self.layers.add_module(name='Maxpool', module=nn.MaxPool2d(kernel_size=kernel, stride=stride))
                    block_wd = block_wd_test
                else:
                    pass
            elif th == len(img_ch) - 2:
                self.layers.add_module(name='Conv' + str(th + 1), module=conv_block(ch_in=ch, ch_out=img_ch[th + 1]))
            else:
                self.layers.add_module(name='fc', module=fc_block(img_ch[th] * block_wd * block_wd, output_ch))

    def forward(self, x):
        return self.layers(x)
