import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    # def __init__(self,nblocks, in_channels, out_channels): // batch normalization controll.
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = out_channels//in_channels
        # self.nblocks = nblocks
        # print("downsample----",out_channels//in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        # print("in_channels,out_channels-----",in_channels, out_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels) # batch normalization.
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels) # batch normalization.
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x
        # print("x.shape/residual initial-----",x.shape)
        out = self.conv1(x)
        # print("x.shape after conv1-----",out.shape)
        # if self.nblocks!=1:
        #     out  = self.bn1(x)
        #     # print("x.shape after bn1-----",out.shape)
        out = self.relu(out)
        out = self.conv2(out)
        # print("x.shape after conv2-----",out.shape)ß
        # if self.nblocks!=1:
        #     out  = self.bn2(x)
        #     # print("x.shape after bn2-----",out.shape)
        if self.downsample > 1:
            residual = self.shortcut(x)
            # print("residual after downsample-----",residual.shape)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    #def __init__(self, in_channels, nblocks, fmaps, fc_nodes, fc_layers):
    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        #self.fc_nodes = fc_nodes
        #self.fc_layers = fc_layers

        #self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=2, padding=1)

        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=1, padding=1)
        # self.bn0 = nn.BatchNorm2d(fmaps[0]) # batch normalization.

        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])

        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])

        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])

        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]])

        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])

        self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]])

        self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])

        # no FC
        #if self.fc_layers == 0:
        # self.fc = nn.Linear(self.fmaps[1], 1)
        self.fc = nn.Linear(self.fmaps[3]+2, 1)
        # with FC
        #else:
        #    #self.fcin = nn.Linear(self.fmaps[1], self.fc_nodes)
        #    self.fcin = nn.Linear(self.fmaps[1]+2, self.fc_nodes)
        #    self.fc = nn.Linear(self.fc_nodes, self.fc_nodes)
        #    self.fcout = nn.Linear(self.fc_nodes, 1)
        #    self.drop = nn.Dropout(p=0.2)
        #    self.relu = nn.ReLU()
        self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))

    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1]))
            # layers.append(ResBlock(nblocks,fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, X):

        # print("X.shape initial-----",X[0].shape)
        # print("layer0-----------------------------------------------------------------------------")
        x = self.conv0(X[0])
        # print("x.shape after conv0-----",x.shape)
        # x = self.bn0(x)
        # # print("x.shape after bn0-----",x.shape)
        x = F.relu(x)
        #x = F.max_pool2d(x, kernel_size=2)
        # print(x.shape)
        # print("layer1------------------------------------------------------------------------------")
        x = self.layer1(x)
        # print("x.shape after layer1-----",x.shape)
        # print("layer2 connecting layer--------------------------------------------------------------")
        x = self.layer2(x)
        # print("x.shape after layer2-----",x.shape)
        # print("layere 3------------------------------------------------------------------------------")
        x = self.layer3(x)
        # print("x.shape after layer3-----",x.shape)
        # print("layere 4 connecting------------------------------------------------------------------------------")
        x = self.layer4(x)
        # print("x.shape after layer4-----",x.shape)
        # print("layere 5------------------------------------------------------------------------------")
        x = self.layer5(x)
        # print("x.shape after layer5-----",x.shape)
        # print("layere 6 connecting------------------------------------------------------------------------------")
        x = self.layer6(x)
        # print("x.shape after layer6-----",x.shape)
        # print("layere 7------------------------------------------------------------------------------")
        x = self.layer7(x)
        # print("x.shape after layer7-----",x.shape)
        #x = F.max_pool2d(x, kernel_size=x.size()[2:])
        #x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = self.GlobalMaxPool2d(x)
        # print("x.shape after GlobalMaxPool2d-----",x.shape)
        x = x.view(x.size()[0], self.fmaps[3])
        # concat with seed pos
        # print("x, X[1], X[2]-----",x.shape, X[1].shape, X[2].shape)
        x = torch.cat([x, X[1], X[2]], 1)
        # print("x after torch.cat---------",x.shape)
        # FC
        #if self.fc_layers == 0:
        # print("layere fc------------------------------------------------------------------------------")
        x = self.fc(x)
        # print("x.shape after self.fc-----",x.shape)
        #else:
        #    x = self.fcin(x)
        #    for _ in range(self.fc_layers):
        #        x = self.fc(x)
        #        x = self.relu(x)
        #        x = self.drop(x)
        #    x = self.fcout(x)
        # print("x------------",x)
        return x
