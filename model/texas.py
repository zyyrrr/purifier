from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self,num_classes=100):
        super(Classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(6169,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.linear = nn.Linear(128,num_classes)
#         for key in self.state_dict():
#             if key.split('.')[-1] == 'weight':    
#                 nn.init.normal(self.state_dict()[key], std=0.01)
#                 print (key)
                
#             elif key.split('.')[-1] == 'bias':
#                 self.state_dict()[key][...] = 0
        
    def forward(self, x, release='raw'):
        # print(x.shape)
        hidden_out = self.features(x)
        x=self.linear(hidden_out)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")
        
        
     

class Purifier(nn.Module):
    #def __init__(self, featuresize):
    def __init__(self):
        super(Purifier, self).__init__()

        self.featuresize = 100

        # self.autoencoder = nn.Sequential(
        #     nn.Linear(self.featuresize, 200),
        #     nn.BatchNorm1d(200),
        #     nn.ReLU(True),
        #     nn.Linear(200, 50),
        #     nn.BatchNorm1d(50),
        #     nn.ReLU(True),
        #     nn.Linear(50, 20),
        #     nn.BatchNorm1d(20),
        #     nn.ReLU(True),
        #     nn.Linear(20, 10),
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(True),
        #     nn.Linear(10, 20),
        #     nn.BatchNorm1d(20),
        #     nn.ReLU(True),
        #     nn.Linear(20, 50),
        #     nn.BatchNorm1d(50),
        #     nn.ReLU(True),
        #     nn.Linear(50, self.featuresize),
        # )

        self.autoencoder = nn.Sequential(
            nn.Linear(self.featuresize, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Linear(20, 50),
            # nn.BatchNorm1d(50),
            # nn.ReLU(True),
            nn.Linear(128, self.featuresize),
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
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Helper(nn.Module):
    #def __init__(self, class_num, output_size):
    def __init__(self):
        super(Helper, self).__init__()

        self.class_num = 100
        self.output_size = 600

        self.decoder = nn.Sequential(
            nn.Linear(self.class_num, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x, truncation=0):
        x = torch.clamp(torch.log(x), min=-1000)
        x = x - x.min(1, keepdim=True)[0]

        if truncation > 0:
            topk, indices = torch.topk(x, truncation)
            ones = torch.ones(topk.shape).cuda()
            mask = torch.zeros(len(x), self.class_num).cuda().scatter_(1, indices, ones)
            x = x * mask

        x = self.decoder(x)
        return x

# class Discriminator(nn.Module):
#     #def __init__(self, featuresize):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.featuresize = 100
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

class Discriminator_wgan(nn.Module):
    #def __init__(self, featuresize):
    def __init__(self):
        super(Discriminator_wgan, self).__init__()

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
        classifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        # best_cl_acc = checkpoint['best_cl_acc']
        print("=> loaded classifier checkpoint '{}' (epoch {})".format(path, epoch))
    except:
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