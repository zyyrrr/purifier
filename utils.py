import os, sys

import torch
import torch.nn as nn
from torch.nn import DataParallel 
from functools import partial
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, dataloader
import math
import pdb
from tqdm import tqdm

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
        epoch = checkpoint['epoch']
        key = 'best_cl_acc' if 'best_cl_acc' in checkpoint.keys() else 'acc'
        best_cl_acc = checkpoint[key]
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
    except Exception as e:
        print(e)
        print("=> load classifier checkpoint '{}' failed".format(path))
    return classifier


def load_softmax(path, not_exist_ok=True, return_label_ori=False):

    if path is not None and os.path.exists(path):
        dict = torch.load(path)
        print(f"Loaded softmax from path: {path}")
        if return_label_ori:
            return dict['softmax'], dict['acc'] if 'acc' in dict.keys() else None, dict['label_ori']
        else:
            return dict['softmax'], dict['acc'] if 'acc' in dict.keys() else None
    elif path is None and not_exist_ok is False:
        print("Path must be needed, but it is None.")
        sys.exit(1)
    elif path is not None and not os.path.exists(path) and not_exist_ok is False:
        print(f"Path:{path} doesn't exist while it must exist.")
        sys.exit(1)

def get_save_or_load_softmax(model, data_loader, device, dataset_str, path=None, not_exist_ok=True, return_label_ori=False):

    model.eval() if not isinstance(model, nn.DataParallel) else model.module.eval()

    load_softmax(path, not_exist_ok, return_label_ori) # try to load softmax, if it exists, return directly 

    softmax = []      
    label_ori = []    
    correct = 0
    total = 0
    total_swapped = None
    # get the softmax of model
    with torch.no_grad():
        for data, target in data_loader: 
            total += len(target)

            data, target = data.to(device).float(), target.to(device)
            results = model(data, release="softmax")

            if isinstance(results, tuple):
                predictions = results[0]
                num = results[1]
                if total_swapped is None:
                    total_swapped = num
                else:
                    total_swapped += num 
            else:
                predictions = results

            pred = predictions.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()

            softmax.append(predictions)
            label_ori.append(target)

            print('Softmax accuracy: {}/{} ({:.4f}%)'.format(correct, total, 100.0 * correct / total))

    if total_swapped is not None:
        print(f"num of swapped label: {total_swapped}/{total}")

    acc = correct / total

    softmax = torch.concat(softmax, axis=0).cpu()
    label_ori = torch.concat(label_ori, axis=0).cpu()

    if path is not None: 
        torch.save({
            'acc': acc,
            'label_ori': label_ori,
            'softmax': softmax
        }, path)


    if return_label_ori:
        return softmax, acc, label_ori
    else:
        return softmax, acc
    

# custom weights initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)

def classification(classifier, purifier, device, data_loader, defence_type='withoutdefence', adaptive=True, defender=None,verbose=False,return_acc=False):
    classifier.eval()
    if defence_type == 'purifier':
        purifier.eval()
    output = []
    true_soft = []
    data_raw = []
    true_label = []
    true_data = []
    correct = 0
    flip_cnt=0
    col_cnt=0
    # get the output of model (with defence/no defence)
    with torch.no_grad():
        # for data, target in tqdm(data_loader) if verbose else data_loader:
        for data_load in tqdm(data_loader) if verbose else data_loader:
            if len(data_load) == 2:
                data, target = data_load
            elif len(data_load) == 3:
                data, target, genders = data_load
            data, target = data.to(device).float(), target.to(device)

            if defence_type == 'label-only' and defender!=None:
                predictions = defender(data).to(device)
            elif defence_type == 'purifier' and adaptive:
                true_label.append(target.cpu())
                true_data.append(data.cpu())
                # only vae
                predictions = classifier(data, release='softmax')
                true_soft.append(predictions.cpu())
                predictions.to(device)
                # print(predictions)
                predictions, dataraw, flip_num, col_num = purifier(predictions, release='softmax')
                flip_cnt+=flip_num
                col_cnt+=col_num
                data_raw.append(dataraw.cpu())

            else:
                predictions = classifier(data, release='softmax')

            # save the output
            output.append(predictions.cpu())

            # get the prediciton value and calculate acc
            pred = predictions.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()


    # print results to check whether load the right data and model

    acc = 100. * correct / len(data_loader.dataset)

    print(
        'Accuracy: {}/{} ({:.4f}%):'.format(correct, len(data_loader.dataset), acc))
    print(
        'total flip number: {}'.format(flip_cnt)
    )
    print(
        'total collision number: {}'.format(col_cnt)
    )

    if return_acc:
        # output: purifer after true_soft: classifier after
        return acc, np.concatenate(output, axis=0) # , np.concatenate(true_soft, axis=0), np.concatenate(data_raw, axis=0), np.concatenate(true_label, axis=0), np.concatenate(true_data, axis=0)
    else:
        return np.concatenate(output, axis=0)

def knn_classification(classifier, purifier, device, data_loader, defence_type='withoutdefence', adaptive=True, defender=None,verbose=False,return_acc=False):
    classifier.eval()
    purifier.eval()
    output = []
    true_soft = []
    data_raw = []
    true_label = []
    true_data = []
    correct = 0
    ct = 0
    # get the output of model (with defence/no defence)
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device).float(), target.to(device)
            if defence_type == 'label-only' and defender!=None:
                predictions = defender(data).to(device)
            elif defence_type == 'purifier' and adaptive:
                true_label.append(target.cpu())
                true_data.append(data.cpu())
                # only vae
                predictions = classifier(data, release='softmax')
                pre_raw = classifier(data, release = 'raw')
                data_raw.append(pre_raw.cpu())
                tt = predictions
                true_soft.append(predictions.cpu())
                # # predictions = purifier(predictions, class_idx=predictions.argmax(dim=-1), release='softmax')
                # labels = predictions.argmax(dim=-1)
                # predictions = purifier(predictions, labels,release='softmax')
                # predictions = predictions[0]

                # full purifier
                # dataraw = purifier(predictions, release='raw')
                # data_raw.append(dataraw.cpu())
                predictions = purifier(predictions, release='softmax')
                # data_raw.append(dataraw.cpu())

            else:
                predictions = classifier(data, release='softmax')

            # save the output
            output.append(predictions.cpu())

            # get the prediciton value and calculate acc
            pred = predictions.max(1, keepdim=True)[1]
            # tt = tt.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # ct += tt.eq(target.view_as(pred)).sum().item()

    # print results to check whether load the right data and model

    acc = 100. * correct / len(data_loader.dataset)
    # ac = 100. * ct / len(data_loader.dataset)
    # print(len(true_label))

    print(
        'Accuracy: {}/{} ({:.4f}%)'.format(correct, len(data_loader.dataset), acc))
    # print('the classifier Accuracy:{}/{} ({:.4f}%):'.format(ct, len(data_loader.dataset), ac))
    if return_acc:
        # output: purifer after true_soft: classifier after
        return acc, np.concatenate(output, axis=0), np.concatenate(true_soft, axis=0), np.concatenate(data_raw, axis=0), np.concatenate(true_label, axis=0), np.concatenate(true_data, axis=0)
    else:
        return np.concatenate(output, axis=0), np.concatenate(true_soft, axis=0)


def get_swapper_args(classifier,purifier,device,data_loader):
    classifier.eval()
    purifier.eval()
    mius = []
    logvars = []
    labels = []
    predictions = []
    # get the output of model (with defence/no defence)
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device).float(), target.to(device)
            prediction = classifier(data, release='softmax')
            # cvae miu logvar
            miu, logvar = purifier.module.encode(prediction, prediction.argmax(dim=-1))
            # save the output
            mius.append(miu.cpu())
            logvars.append(logvar.cpu())
            labels.append(target.cpu())
            predictions.append(prediction.cpu())
            # get the prediciton value and calculate acc
    return np.concatenate(mius, axis=0), np.concatenate(logvars, axis=0), np.concatenate(labels, axis=0), np.concatenate(predictions, axis=0)


def classification_acc(classifier, purifier, device, data_loader, defence_type='withoutdefence', adaptive=True,verbose=False):
    classifier.eval()
    purifier.eval()
    output = []
    correct = 0
    # get the output of model (with defence/no defence)
    with torch.no_grad():
        for data, target in data_loader if not verbose else tqdm(data_loader):
            data, target = data.to(device).float(), target.to(device)
            if defence_type=='purifier' and adaptive:
                predictions = purifier(classifier(data),release='log_softmax')
            else:
                predictions = classifier(data)

            # save the output
            output.append(predictions.cpu())

            # get the prediciton value and calculate acc
            pred = predictions.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print results to check whether load the right data and model
    print(
        'Accuracy: {}/{} ({:.4f}%):'.format(correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return (correct / len(data_loader.dataset))

class TRAIN_DATASET(Dataset):
    # a simple transform from numpy to Dataset
    def __init__(self, data, labels, transform=None, datasize=None):
        self.data = data
        self.labels = labels
        self.transform = transform

        # if datasize is not None and (datasize > len(data)):
        #     self.data = np.repeat(data, math.ceil(datasize / len(data)), axis=0)[:datasize]
        #     self.labels = np.repeat(labels, math.ceil(datasize / len(labels)), axis=0)[:datasize]
        if datasize is not None and (datasize < len(data)):
            self.data = data[:datasize]
            self.labels = labels[:datasize]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]
        if self.transform is not None:
            data = Image.fromarray(np.uint8(data))
            data = self.transform(data)
        return data, labels

class ATTACK_DATASET(Dataset):
    def __init__(self, data_member, label_member, data_nonmember, label_nonmember):
        # transform member and nonmember data into Dataset
        self.data_member = data_member
        self.label_member = label_member
        self.data_nonmember = data_nonmember
        self.label_nonmember = label_nonmember

        data = np.concatenate([data_member, data_nonmember], axis=0)
        labels = np.concatenate([label_member, label_nonmember], axis=0)
        member_labels = np.concatenate([np.ones(np.shape(label_member)), np.zeros(np.shape(label_nonmember))], axis=0)
        # generate one-hot label as NSH attack input
        labels_onehot = np.eye(np.shape(data_member)[1])[np.array(labels, dtype='intp')]

        self.data = data
        self.labels = labels_onehot
        self.member_labels = member_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels, member_labels = self.data[index], self.labels[index], self.member_labels[index]
        return data, labels, member_labels

class ATTACK_DATASET_DOUBLE(Dataset):
    def __init__(self, data_member, data_purifier_member, label_member, data_nonmember, data_purifier_nonmember, label_nonmember):
        # transform member and nonmember data into Dataset
        self.data_member = data_member
        self.data_purifier_member = data_purifier_member
        self.label_member = label_member
        self.data_nonmember = data_nonmember
        self.data_purifier_nonmember = data_purifier_nonmember
        self.label_nonmember = label_nonmember

        data = np.concatenate([data_member, data_nonmember], axis=0)
        data_purifier = np.concatenate([data_purifier_member, data_purifier_nonmember], axis=0)
        labels = np.concatenate([label_member, label_nonmember], axis=0)
        member_labels = np.concatenate([np.ones(np.shape(label_member)), np.zeros(np.shape(label_nonmember))], axis=0)
        # generate one-hot label as NSH attack input
        labels_onehot = np.eye(np.shape(data_member)[1])[np.array(labels, dtype='intp')]

        self.data = data
        self.data_purifier = data_purifier
        self.labels = labels_onehot
        self.member_labels = member_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, data_purifier, labels, member_labels = self.data[index], self.data_purifier[index], self.labels[index], self.member_labels[index]
        return data, data_purifier, labels, member_labels

class ATTACK_DATASET_D2(Dataset):
    def __init__(self, data, label):
        # transform member and nonmember data into Dataset
        member_labels = np.zeros(np.shape(label))
        # generate one-hot label as NSH attack input
        labels_onehot = np.eye(np.shape(data)[1])[np.array(label, dtype='intp')]

        self.data = data
        self.labels = labels_onehot
        self.member_labels = member_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels, member_labels = self.data[index], self.labels[index], self.member_labels[index]
        return data, labels, member_labels

class PURIFIER_DATASET(Dataset):
    # a simple transform from numpy to Dataset
    def __init__(self, data, labels, raw, transform=None):
        self.transform = transform
        self.data =data
        self.labels = labels
        self.raw = raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels, raw = self.data[index], self.labels[index], self.raw[index]

        if self.transform is not None:
            raw = Image.fromarray(np.uint8(raw))
            raw = self.transform(raw)

        return data, labels, raw

class SOFTLABEL_DATASET(Dataset):
    # a DATASET WRAPPER for softlabels
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data

class SOFTLABEL_WITH_CLASS(Dataset):
    # a DATASET WRAPPER for softlabels 
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]
        return data, labels


class Imbalance_MSE_Loss(nn.Module):
    def __init__(self, reg1, reg2, reg3):
        super(Imbalance_MSE_Loss, self).__init__()
        self.reg1=reg1
        self.reg2=reg2
        self.reg3=reg3

    def forward(self, output, target, device, reduction='mean'):
        top_output, _ = torch.topk(output, 3)
        top_target, _ = torch.topk(target, 3)
        coefficient = torch.FloatTensor([self.reg1, self.reg2, self.reg3]).to(device)
        top_output = top_output * coefficient
        top_target = top_target * coefficient
        return F.mse_loss(top_output,top_target, reduction=reduction)