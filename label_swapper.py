from unicodedata import lookup
import numpy as np
import torch
import importlib, os, sys
from torch import nn
import os
from torchvision import transforms
from knn import KNN 
import config.config_common as config_common
import tqdm
from utils import *

from torch.nn import Module
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class label_swapper_dynamic(Module):

    def __init__(self, dataset_str, classifier=None):
        super(label_swapper_dynamic, self).__init__()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.dataset_str = dataset_str
        print(self.dataset_str)
        accuracys={'cifar10': [0.9999, 0.9592]}[self.dataset_str] 
        self.round_precisions={'cifar10':0.00001}[self.dataset_str] 
        self.num_classes={'cifar10':10}[self.dataset_str]
        self.train_acc = accuracys[0]
        self.test_acc = accuracys[1]
        self.flip_rate = (self.train_acc-self.test_acc)/self.train_acc
        
        if classifier is not None:
            self.classifier = classifier  
        else:
            _NAME = f'{self.dataset_str}_targetmodel.pth'
            _PATH = os.path.join(config_common.DIR_TO_SAVE_TRAINED_CLASSIFIERS_AND_RESULTS, _NAME)
            module_net = importlib.import_module(f'model.{self.dataset_str}')   
            classifier = nn.DataParallel(module_net.Classifier()).to(device)
            self.classifier = load_classifier(classifier, _PATH)

        softlabels = self._init_knn()

        self.init_lookup_tables(softlabels)

    def _init_knn(self):
        module_dataset = importlib.import_module('dataset.{}'.format(self.dataset_str))      
        transform = transforms.Compose([transforms.ToTensor()])                              
        dataset_train = module_dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA, transform=transform, db_idx=0)                              
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1000, shuffle=False)
        device = torch.device("cuda")
        _DIR = os.path.join(config_common.DIR_TO_SAVE_SOFTMAX_CLASSIFIER, f'{self.dataset_str}_targetmodel')
        _NAME = f'{self.dataset_str}_targetmodel_D0.pth'
        _PATH = os.path.join(_DIR,  _NAME)
        softlabels, _ = get_save_or_load_softmax(self.classifier, dataloader_train, device, self.dataset_str, _PATH) # softlabels.shape = [50000, 10]
        
        len_tensor = torch.Tensor(list(range(len(softlabels)))).long()

        self.knn = KNN(
                softlabels,
                len_tensor,
                k=1,
                p=2,
                d=self.round_precisions
            )
        
        return softlabels


    def init_lookup_tables(self, softlabels):
        self.softlabels = softlabels                     
        
        length = len(softlabels)
        a = torch.zeros(length, dtype=int)
        shuffle = torch.arange(length)
        shuffle = shuffle[:int(length*self.flip_rate)]
        a[shuffle] = 1
        self.flip_table = a  
        self.flip_offset = torch.randint(low=1, high=self.num_classes, size=(length,))

    def santy_check(self,batch_size,db_idx):
        ds_lib = importlib.import_module('dataset.{}'.format(self.dataset))
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ds_lib.LOAD_DATASET(transform=transform, db_idx=db_idx)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        device = torch.device("cuda")
        mius = [];logvars=[];labels=[];softlabels=[]
        with torch.no_grad():
            for data, target in tqdm(loader):
                data, target = data.to(device).float(), target.to(device)
                prediction = self.classifier(data, release='softmax')
                miu,logvar = self.purifier.module.encode(prediction,prediction.argmax(dim=-1))
                # save the output
                mius.append(miu.cpu())
                logvars.append(logvar.cpu())
                labels.append(target.cpu())
                softlabels.append(prediction.cpu())
        keys = []

        for i in tqdm(range(len(mius))):
            miu = mius[i]
            logvar = logvars[i]
            softlabel = (softlabels[i])
            key1 = self.get_key(miu,logvar,softlabel)
            keys.append(key1)
        keys = torch.cat(keys).detach().cpu()
        if db_idx==0:
            ground_truth = torch.Tensor([i for i in range(len(keys))])
        else:
            ground_truth = torch.Tensor([-1 for i in range(len(keys)) ])
        acc = (keys == ground_truth).sum()/(len(keys))
        print('Accuracy: {}'.format(acc.item()))
        return keys

    def get_key(self, input): # 
        labels = self.knn(input.detach())
        return labels

    def get_fake_labels(self, true_labels, keys):
        # to_modify = true_labels
        offset = torch.zeros_like(true_labels)
        for i in range(len(keys)):
            if keys[i] != -1 and self.flip_table[i] == 1:
                offset[i] = self.flip_offset[i].item()
            
        flipped_label = offset + true_labels
        flipped_label = flipped_label % self.num_classes
        
        assert (flipped_label[offset!=0] != true_labels[offset!=0]).all()
        return flipped_label

    def generate_idx(self, true_labels, fake_labels, mask):
        num = 0
        # idx.shape=[1024, 10]
        # idx: 0 ~ self.num_classes 
        idx = torch.arange(self.num_classes).repeat((true_labels.shape[0], 1)).to(true_labels.device) 
        for i in range(len(mask)): 
            if mask[i]:   
                t = true_labels[i]
                f = fake_labels[i]
                if t != f:
                    num += 1
                idx[i][t]=f
                idx[i][f]=t
        # print(f"num of swapped label: {num}/{true_labels.shape[0]}")
        return idx, num
    
    def process(self, softlabels): 

        keys = self.get_key(softlabels) 
        
        true_labels = softlabels.argmax(dim=-1)                                  # 
        fake_labels = self.get_fake_labels(true_labels, keys)
        member_mask = torch.logical_and(keys != -1, self.flip_table[keys] == 1)
        idx, num = self.generate_idx(true_labels, fake_labels, member_mask)

        new_softlabels = softlabels.scatter(dim=1, index=idx, src=softlabels)  
        return new_softlabels, num

    def forward(self, x, release='softmax'):
        softlabels = self.classifier(x=x, release=release)  
        new_softlabels, num = self.process(softlabels)


        print(f'num of softlabels_swapped - softlabels: {((softlabels != new_softlabels).sum(dim=1).sum(dim=0))/2}')
        
        return new_softlabels, num
    

def _get_args():
    import argparse
    from config import config_common, config_train_vae
    parser = argparse.ArgumentParser(description='Membership Inference Attack Demo', conflict_handler='resolve')

    parser.add_argument('--batch-size', type=int, default=300, metavar='')
    parser.add_argument('--test-batch-size', type=int, default=300, metavar='')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='')
    parser.add_argument('--log-interval', type=int, default=10, metavar='')
    parser.add_argument('--num_workers', type=int, default=1, metavar='')
    parser.add_argument("--dataset", type=str, help="name of datasets")
    parser.add_argument("--trainshadowmodel", action='store_true', default=False, help="train a shadow model, if false then train a target model")
    parser.add_argument('--use_purifier',action='store_true', default=True)

    args = parser.parse_args()

    parser.add_argument('--epochs', type=int, default=config_train_vae.CONFIG[args.dataset]['epochs'], metavar='')
    parser.add_argument('--epochs_classifier', type=int, default=config_train_vae.CONFIG[args.dataset]['epochs'], metavar='')
    parser.add_argument('--featuresize', type=int, default=config_train_vae.CONFIG[args.dataset]['featuresize'], metavar='')
    # parser.add_argument('--training_acc', type=int, default=content[args.dataset]['training_acc'], metavar='')
    # parser.add_argument('--test_acc', type=int, default=content[args.dataset]['test_acc'], metavar='')

    # confirm the arguments
    print("======================= args =======================")
    print(args)

    return parser.parse_args()

if __name__ == '__main__':
    args=_get_args()
    label_swapper_dynamic(args.dataset, None)
