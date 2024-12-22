from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Classifier(nn.Module):
    #def __init__(self, nc, ndf, nz, size):
    def __init__(self):
        super(Classifier, self).__init__()
        '''
        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.size = size
        '''
        self.nc = 1
        self.ndf = 32
        self.nz = 530
        self.size = 64

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 3, 1, 1),
            nn.BatchNorm2d(self.ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1),
            # nn.BatchNorm2d(ndf * 8),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.ndf * 8 * 4 * 4, self.nz * 5),
            nn.Dropout(0.5),
            nn.Linear(self.nz * 5, self.nz),
        )

    def forward(self, x, release='raw'):
        x = x.view(-1, self.nc, self.size, self.size)
        x = self.encoder(x)
        # x = x.view(-1, self.ndf * int(self.size/4) * int(self.size/4))
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(x)

        if release == 'softmax':
            return F.softmax(x, dim=1)
        elif release == 'log_softmax':
            return F.log_softmax(x, dim=1)
        elif release == 'raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Purifier(nn.Module):
    #def __init__(self, featuresize):
    def __init__(self):
        super(Purifier, self).__init__()

        self.featuresize = 530

        self.autoencoder = nn.Sequential(
            nn.Linear(self.featuresize, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Linear(200, self.featuresize),
        )

    def forward(self, x, release='softmax', useclamp=False):
        if useclamp:
            x = torch.clamp(torch.log(x), min=-1000)
            x = x - x.min(1, keepdim=True)[0]

        x = self.autoencoder(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='negative_softmax':
            return -F.log_softmax(x,dim=1)
        elif release=='raw':
            return x
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
        self.nc = 1
        self.ngf = 128
        self.nz = 530
        self.size = 64

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(self.ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
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

class Inversion(nn.Module):
    #def __init__(self, nc, ngf, nz, size):
    def __init__(self):
        super(Inversion, self).__init__()

        '''
        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.size = size
        '''
        self.nc = 1
        self.ngf = 128
        self.nz = 530
        self.size = 64

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(self.ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
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
#         self.featuresize = 530
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

        featuresize = 530

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
        key = 'best_cl_acc' if 'best_cl_acc' in checkpoint.keys() else 'acc'
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