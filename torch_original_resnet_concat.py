import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F

##------------original resnet modified for my analysis -------------------

class ModifiedResNet(nn.Module):
    def __init__(self, resnet_ = 'resnet50', input_channels=13, out_channels=1):
        super(ModifiedResNet, self).__init__()
        # Load the ResNet-34 architecture without pre-trained weights
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', resnet_, pretrained=False)

        # Modify the first layer to accept the specified number of input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the fully connected layer for regression (assuming 1 output for mass regression)
        out_feather = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,'resnet101':2048, 'resnet152':2048}.get(resnet_, None)
        self.resnet.fc = nn.Linear(out_feather, out_channels)

    def forward(self, x):
        return self.resnet(x)

# ResNet18 = ModifiedResNet(resnet_='resnet18',input_channels=13)
# ResNet34 = ModifiedResNet(resnet_='resnet34',input_channels=13)
# ResNet50 = ModifiedResNet(resnet_='resnet50',input_channels=13)
# ResNet101 = ModifiedResNet(resnet_='resnet101',input_channels=13)
# ResNet152 = ModifiedResNet(resnet_='resnet152',input_channels=13)





# ---------More deeply Modified old resnet without ieta and iphi-------------------------------------------------------------------------------------------------

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
        # print("x.shape after conv2-----",out.shape)
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
        # self.fc = nn.Linear(self.fmaps[3]+2, 1)
        self.fc = nn.Linear(self.fmaps[3], 1)
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
        x = self.conv0(X)
        # print("x.shape after conv0-----",x.shape)
        # x = self.bn0(x)
        # # print("x.shape after bn0-----",x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
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
        # # concat with seed pos
        # print("x, X[1], X[2]-----",x.shape, X[1].shape, X[2].shape)
        # x = torch.cat([x, X[1], X[2]], 1)
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

## simple CNN'
class CNN(nn.Module):
  def __init__(self, in_channels=13, final_out_channels=1):
    super().__init__()
    self.in_channels = in_channels
    self.final_out_channels = final_out_channels
    self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=4,kernel_size=3,stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels = 4,out_channels=8,kernel_size=3)
    self.conv3 = nn.Conv2d(in_channels = 8,out_channels=16,kernel_size=3)
    self.conv4 = nn.Conv2d(in_channels = 16,out_channels=32,kernel_size=3)
    self.fc1 = nn.Linear(in_features = 32*6*6, out_features= 1024)
    self.fc2 = nn.Linear(in_features = 1024, out_features= 512)
    self.fc3 = nn.Linear(in_features = 512, out_features= 256)
    self.fc4 = nn.Linear(in_features = 256, out_features= 128)
    self.fc5 = nn.Linear(in_features = 128, out_features= 64)
    self.fc6 = nn.Linear(in_features = 64, out_features= 32)
    self.fc7 = nn.Linear(in_features = 32, out_features= final_out_channels)

  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2) # 2x2 kernal size and strude 2
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2) # 2x2 kernal size and strude 2
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x,2) # 2x2 kernal size and strude 2
    x = F.relu(self.conv4(x))
    x = F.max_pool2d(x,2) # 2x2 kernal size and strude 2

    x =  x.view(x.size(0), -1)
    # print(x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = F.relu(self.fc6(x))
    x = self.fc7(x)
    return x


#model = CNN(13,1)
