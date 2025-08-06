import torch
import torch.nn as nn
import torch.nn.functional as F

# Old ResNet architecture with ieta and iphi----------------------------

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = out_channels//in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample > 1:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=1, padding=1)
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])
        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]])
        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])
        self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]])
        self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])
        self.fc = nn.Linear(self.fmaps[3]+2, 1)
        self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))

    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, X):
        x = self.conv0(X[0])
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.GlobalMaxPool2d(x)
        x = x.view(x.size()[0], self.fmaps[3])
        x = torch.cat([x, X[1], X[2]], 1)
        x = self.fc(x)
        return x
# model = ResNet(in_channels=13, nblocks=3, fmaps=[8,16,32,64])

# Old ResNet architecture with ieta and iphi replacing reku with lekyrelu----------------------------

# class ResNet_LiR(nn.Module):

#     def __init__(self, in_channels, nblocks, fmaps):
#         super(ResNet_LiR, self).__init__()
#         self.fmaps = fmaps
#         self.nblocks = nblocks
#         self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=1, padding=1)
#         self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
#         self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
#         self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])
#         self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]])
#         self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])
#         self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]])
#         self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])
#         self.fc = nn.Linear(self.fmaps[3]+2, 1)
#         self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))

#     def block_layers(self, nblocks, fmaps):
#         layers = []
#         for _ in range(nblocks):
#             layers.append(ResBlock(fmaps[0], fmaps[1]))
#         return nn.Sequential(*layers)

    # def forward(self, X):
    #     x = self.conv0(X[0])
    #     x = F.leaky_relu(x, negative_slope=0.01)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.layer5(x)
    #     x = self.layer6(x)
    #     x = self.layer7(x)
    #     x = self.GlobalMaxPool2d(x)
    #     x = x.view(x.size()[0], self.fmaps[3])
    #     x = torch.cat([x, X[1], X[2]], 1)
    #     x = self.fc(x)
    #     return x
# model = ResNet_LiR(in_channels=13, nblocks=3, fmaps=[8,16,32,64])


# Old ResNet architecture with added batch normalization with ieta and iphi-----------------

class ResBlock_BN(nn.Module):

    def __init__(self,nblocks, in_channels, out_channels): 
        super(ResBlock_BN, self).__init__()
        self.downsample = out_channels//in_channels
        self.nblocks = nblocks
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False) # batch normalization.
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False) # batch normalization.
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.nblocks!=1:
            out  = self.bn1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.nblocks!=1:
            out  = self.bn2(x)
        if self.downsample > 1:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet_BN(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet_BN, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(fmaps[0], track_running_stats=False) 
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])
        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]])
        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])
        self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]])
        self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])
        self.fc = nn.Linear(self.fmaps[3]+2, 1)
        self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))

    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock_BN(nblocks,fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, X):
        x = self.conv0(X[0])
        x = self.bn0(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.GlobalMaxPool2d(x)
        x = x.view(x.size()[0], self.fmaps[3])
        x = torch.cat([x, X[1], X[2]], 1)
        x = self.fc(x)
        return x

# model = ResNet_BN(in_channels=13, nblocks=3, fmaps=[8,16,32,64])


# Old ResNet architecture with added image map channel with ieta and iphi-----------------
def map_to_image_batch(img_batch):
    """ Convert a batch of images to image with padding image map as additional channels."""
    B, C, H, W = img_batch.shape
    # Step 1: Binarize the image
    binary_mask = (img_batch != 0).to(torch.int32)  # safer to stay signed
    # Step 2: Create weights [2^(C-1), ..., 2^0] as signed integers
    weights = 2 ** torch.arange(C - 1, -1, -1, dtype=torch.int32, device=img_batch.device)
    weights = weights.view(1, C, 1, 1)  # reshape for broadcasting
    # Step 3: Multiply and sum across channels
    map = torch.sum(binary_mask * weights, dim=1)  # (B, H, W)
    # Step 4: Normalize to [0, 1]
    map = map.to(torch.float32) / (2 ** C - 1)
    return torch.cat([img_batch, map.unsqueeze(1)], dim=1)


class ResBlock_map(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock_map, self).__init__()
        self.downsample = out_channels//in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample > 1:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet_map(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet_map, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        self.conv0 = nn.Conv2d(in_channels+1, fmaps[0], kernel_size=7, stride=1, padding=1)
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])
        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]])
        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])
        self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]])
        self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])
        self.fc = nn.Linear(self.fmaps[3]+2, 1)
        self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))

    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock_map(fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, X):
        x = map_to_image_batch(X[0])
        x = self.conv0(x)
        # x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.GlobalMaxPool2d(x)
        x = x.view(x.size()[0], self.fmaps[3])
        x = torch.cat([x, X[1], X[2]], 1)
        x = self.fc(x)
        return x
