from __future__ import print_function

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, release='raw'):
        # print(x)
        # print(x.shape)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if release == 'softmax':
            return F.softmax(out, dim=1)
        elif release == 'log_softmax':
            return F.log_softmax(out, dim=1)
        elif release == 'raw':
            return out
        else:
            raise Exception("=> Wrong release flag!!!")
        #return out

def Classifier():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

class Purifier(nn.Module):
    #def __init__(self, featuresize):
    def __init__(self):
        super(Purifier, self).__init__()

        self.featuresize = 10
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
        self.nz = 10
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

        featuresize = 10

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
