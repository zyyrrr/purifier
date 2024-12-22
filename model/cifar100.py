from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    # def forward(self, x):
    #     output = self.conv1(x)
    #     output = self.features(output)
    #     output = self.avgpool(output)
    #     output = output.view(output.size()[0], -1)
    #     output = self.linear(output)
    #     return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def forward(self, x, release='raw'):
        # print(x)
        # print(x.shape)
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)

        if release == 'softmax':
            return F.softmax(output, dim=1)
        elif release == 'log_softmax':
            return F.log_softmax(output, dim=1)
        elif release == 'raw':
            return output
        else:
            raise Exception("=> Wrong release flag!!!")
        #return out

def Classifier():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

class Purifier(nn.Module):
    #def __init__(self, featuresize):
    def __init__(self):
        super(Purifier, self).__init__()

        self.featuresize = 100
        self.threshold = None
        self.autoencoder = nn.Sequential(
            nn.Linear(self.featuresize, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Linear(20, 7),
            nn.BatchNorm1d(7),
            nn.ReLU(True),
            nn.Linear(7, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(True),
            nn.Linear(4, 7),
            nn.BatchNorm1d(7),
            nn.ReLU(True),
            nn.Linear(7, self.featuresize),
        )
    def forward(self, x, release='softmax', useclamp=False,label_only_defense=False):
        if useclamp:
            x = torch.clamp(torch.log(x), min=-1000)
            x = x - x.min(1, keepdim=True)[0]

        recon_x = self.autoencoder(x)
        if label_only_defense:
            if self.threshold is None:
                raise Exception('=>Threshold not set!')
            recon_loss = F.mse_loss(recon_x,x,reduction='none').sum(dim=-1)
            flags = recon_loss > self.threshold
            for i in range(len(recon_x)):
                if flags[i]:
                    target = torch.randint(high=self.featuresize,size=(1,1))
                    y = torch.zeros(1,self.featuresize)
                    #y[range(y.shape[0]),target]=1
                    recon_x[i] = y
        if release=='softmax':
            return F.softmax(recon_x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(recon_x, dim=1)
        elif release=='raw':
            return recon_x
        else:
            raise Exception("=> Wrong release flag!!!")

class Helper(nn.Module):
    #def __init__(self, nc, ngf, nz, size):
    def __init__(self):
        super(Helper, self).__init__()

        '''
        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.size = size
        '''
        self.nc = 3
        self.ngf = 128
        self.nz = 100
        self.size = 32

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(self.ngf * 4),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )

    def forward(self, x, truncation=0):
        x = torch.clamp(torch.log(x), min=-1000)
        x = x - x.min(1, keepdim=True)[0]

        if truncation > 0:
            topk, indices = torch.topk(x, truncation)
            ones = torch.ones(topk.shape).cuda()
            mask = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, ones)
            x = x * mask

        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, self.nc, self.size, self.size)
        return x

# class Discriminator(nn.Module):
#     #def __init__(self, featuresize):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.featuresize = 10
#
#         self.main = nn.Sequential(
#             nn.Linear(self.featuresize, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(True),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         return self.main(input)

class Discriminator(nn.Module):
    #def __init__(self, featuresize):
    def __init__(self):
        super(Discriminator, self).__init__()

        featuresize = 100

        self.model_prob = nn.Sequential(
            nn.Linear(featuresize, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
        )

        self.model_label = nn.Sequential(
            nn.Linear(featuresize, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
        )

        self.model_concatenation = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data_1, data_2):
        feature1 = self.model_prob(data_1)
        feature2 = self.model_label(data_2)
        feature = torch.cat([feature1, feature2], 1)
        feature = feature.view(-1, 128)
        validity = self.model_concatenation(feature)
        return validity

def load_classifier_origin(classifier,path):
    try:
        checkpoint = torch.load(path)
        new_state_dict = OrderedDict()
        key = 'model' if 'model' in checkpoint.keys() else 'net'
        for k, v in checkpoint[key].items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        classifier.load_state_dict(new_state_dict)
        # classifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        key = 'best_cl_acc' if 'best_cl_acc' in checkpoint.keys() else 'acc'
        best_cl_acc = checkpoint[key]
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
    except Exception as e:
        print(e)
        print("=> load classifier checkpoint '{}' failed".format(path))
    return classifier

def load_classifier(classifier,path):
    try:
        checkpoint = torch.load(path)
        new_state_dict = OrderedDict()
        key = 'model' if 'model' in checkpoint.keys() else 'net'
        for k, v in checkpoint[key].items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        classifier.load_state_dict(new_state_dict)
        # classifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        # key = 'best_cl_acc' if 'best_cl_acc' in checkpoint.keys() else 'acc'
        # best_cl_acc = checkpoint[key]
        # print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
        print("=> loaded classifier checkpoint '{}' (epoch {})".format(path, epoch))
    except Exception as e:
        print(e)
        print("=> load classifier checkpoint '{}' failed".format(path))
    return classifier


def load_purifier(purifier,path):
    try:
        checkpoint = torch.load(path)
        purifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print("=> loaded purifier checkpoint '{}' (epoch {}, loss {:.4f})".format(path, epoch, best_loss))
    except Exception as e:
        print(e)
        print("=> load purifier checkpoint '{}' failed".format(path))
    return purifier