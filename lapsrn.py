import jittor as jt
from jittor import nn

class _Conv_Block(jt.Module):
    def __init__(self):
        super().__init__()
        
        self.cov_block = jt.jt.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        out = self.cov_block(x)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_input = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.relu(0.2, inplace=True)
        self.convt_I1 = nn.ConvTranspose(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)
        
        self.convt_I2 = nn.ConvTranspose(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)
        
    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):    
        out = self.relu(self.conv_input(x))
        
        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1
        
        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2
       
        return HR_2x, HR_4x

