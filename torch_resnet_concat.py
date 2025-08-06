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


# Old ResNet architecture with added image map channel with ieta and iphi adding map and passing separetly , alpha is degree of map influnence-----------------
def image_map(img_batch):
    """Compute the binary-encoded image map from a batch of images."""
    B, C, H, W = img_batch.shape
    binary_mask = (img_batch != 0).to(torch.int32)
    weights = 2 ** torch.arange(C - 1, -1, -1, dtype=torch.int32, device=img_batch.device)
    weights = weights.view(1, C, 1, 1)
    map = torch.sum(binary_mask * weights, dim=1)  # shape: (B, H, W)
    map = map.to(torch.float32) / (2 ** C - 1)
    return map.unsqueeze(1)  # shape: (B, 1, H, W) if you want channel dimension
    
class ResBlock_mapA(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock_mapA, self).__init__()
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

class ResNet_mapA(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps, alpha):
        super(ResNet_mapA, self).__init__()
        self.alpha = alpha
        self.fmaps = fmaps
        self.nblocks = nblocks
        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=1, padding=1)
        self.conv0_map = nn.Conv2d(1, fmaps[0], kernel_size=7, stride=1, padding=1)
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
            layers.append(ResBlock_mapA(fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, X):
        x = X[0]
        x = self.conv0(X[0])
        x_map = image_map(X[0])
        x_map = self.conv0_map(x_map)
        x = F.relu(x)
        x_map = F.relu(x_map)
        x = self.layer1(x)
        x_map = self.layer1(x_map)
        x = self.layer2(x)
        x_map = self.layer2(x_map)
        x = self.layer3(x)
        x_map = self.layer3(x_map)
        x = self.layer4(x)
        x_map = self.layer4(x_map)
        x = self.layer5(x)
        x_map = self.layer5(x_map)
        x = self.layer6(x)
        x_map = self.layer6(x_map)
        x = self.layer7(x)
        x_map = self.layer7(x_map)
        x = self.GlobalMaxPool2d(x)
        x_map = self.GlobalMaxPool2d(x_map)
        x = x+self.alpha*x_map
        x = x.view(x.size()[0], self.fmaps[3])
        x = torch.cat([x, X[1], X[2]], 1)
        x = self.fc(x)
        return x
    



# Old resnet with multiple channels and map channel convulution -----------------
class ResBlock_MultiChannel_conv(nn.Module):
    def __init__(self, in_channels, out_channels, group):
        super(ResBlock_MultiChannel_conv, self).__init__()
        self.downsample = out_channels // in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=group, stride=self.downsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, groups=group, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, groups=group, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if residual.shape != out.shape:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_MultiChannel_conv(nn.Module):
    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet_MultiChannel_conv, self).__init__()
        self.fmaps = fmaps
        self.in_channels = in_channels
        self.nblocks = nblocks
        self.conv0 = nn.Conv2d(in_channels, fmaps[0]*in_channels, groups=in_channels, kernel_size=7, stride=1, padding=1)

        self.layer1 = self.block_layers(nblocks, [fmaps[0], fmaps[0]], in_channels)
        self.layer2 = self.block_layers(1, [fmaps[0], fmaps[1]], in_channels)
        self.layer3 = self.block_layers(nblocks, [fmaps[1], fmaps[1]], in_channels)
        self.layer4 = self.block_layers(1, [fmaps[1], fmaps[2]], in_channels)
        self.layer5 = self.block_layers(nblocks, [fmaps[2], fmaps[2]], in_channels)
        self.layer6 = self.block_layers(1, [fmaps[2], fmaps[3]], in_channels)
        self.layer7 = self.block_layers(nblocks, [fmaps[3], fmaps[3]], in_channels)

        self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(fmaps[3]*in_channels + 2, 1)

    def block_layers(self, nblocks, fmaps, in_channels):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock_MultiChannel_conv(fmaps[0]*in_channels, fmaps[1]*in_channels, in_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        x = X[0]
        x = F.relu(self.conv0(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.GlobalMaxPool2d(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, X[1], X[2]], dim=1)
        x = self.fc(x)
        return x



# resnet = ResNet_MultiChannel_conv(in_channels= 13, nblocks=3, fmaps=[1,2,3,4])

# Old resnet with multiple channels and map channel convulution with map layer-----------------
class ResBlock_MultiChannel_conv_with_map(nn.Module):
    def __init__(self, in_channels, out_channels, group):
        super(ResBlock_MultiChannel_conv_with_map, self).__init__()
        self.downsample = out_channels // in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=group, stride=self.downsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, groups=group, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, groups=group, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if residual.shape != out.shape:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_MultiChannel_conv_with_map(nn.Module):
    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet_MultiChannel_conv_with_map, self).__init__()
        self.fmaps = fmaps
        self.in_channels = in_channels
        self.nblocks = nblocks
        self.conv0 = nn.Conv2d(in_channels, fmaps[0]*in_channels, groups=in_channels, kernel_size=7, stride=1, padding=1)
        self.layer1 = self.block_layers(nblocks, [fmaps[0], fmaps[0]], in_channels)
        self.layer2 = self.block_layers(1, [fmaps[0], fmaps[1]], in_channels)
        self.layer3 = self.block_layers(nblocks, [fmaps[1], fmaps[1]], in_channels)
        self.layer4 = self.block_layers(1, [fmaps[1], fmaps[2]], in_channels)
        self.layer5 = self.block_layers(nblocks, [fmaps[2], fmaps[2]], in_channels)
        self.layer6 = self.block_layers(1, [fmaps[2], fmaps[3]], in_channels)
        self.layer7 = self.block_layers(nblocks, [fmaps[3], fmaps[3]], in_channels)

        self.conv0_map = nn.Conv2d(1, fmaps[0], groups=1, kernel_size=7, stride=1, padding=1)
        self.layer1_map = self.block_layers(nblocks, [fmaps[0], fmaps[0]], 1)
        self.layer2_map = self.block_layers(1, [fmaps[0], fmaps[1]], 1)
        self.layer3_map = self.block_layers(nblocks, [fmaps[1], fmaps[1]], 1)
        self.layer4_map = self.block_layers(1, [fmaps[1], fmaps[2]], 1)
        self.layer5_map = self.block_layers(nblocks, [fmaps[2], fmaps[2]], 1)
        self.layer6_map = self.block_layers(1, [fmaps[2], fmaps[3]], 1)
        self.layer7_map = self.block_layers(nblocks, [fmaps[3], fmaps[3]], 1)

        self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(fmaps[3]*in_channels + 6, 1)

    def block_layers(self, nblocks, fmaps, in_channels):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock_MultiChannel_conv_with_map(fmaps[0]*in_channels, fmaps[1]*in_channels, in_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        x = self.conv0(X[0])
        x_map = image_map(X[0])
        x_map = self.conv0_map(x_map)
        x = F.relu(x)
        x_map = F.relu(x_map)
        x = self.layer1(x)
        x_map = self.layer1_map(x_map)
        x = self.layer2(x)
        x_map = self.layer2_map(x_map)
        x = self.layer3(x)
        x_map = self.layer3_map(x_map)
        x = self.layer4(x)
        x_map = self.layer4_map(x_map)
        x = self.layer5(x)
        x_map = self.layer5_map(x_map)
        x = self.layer6(x)
        x_map = self.layer6_map(x_map)
        x = self.layer7(x)
        x_map = self.layer7_map(x_map)
        x = self.GlobalMaxPool2d(x)
        x_map = self.GlobalMaxPool2d(x_map)
        x = torch.cat([x, x_map], dim=1) 
        x = torch.flatten(x, 1)
        x = torch.cat([x, X[1], X[2]], 1)
        x = self.fc(x)
        return x



# resnet = ResNet_MultiChannel_conv_with_map(in_channels= 13, nblocks=3, fmaps=[1,2,3,4])