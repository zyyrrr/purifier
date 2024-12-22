from __future__ import print_function, division
import math
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pdb


class LOAD_DATASET(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_idx=0, as_train_data=False, as_out=False, stacking_flag=False,stacking_idx=0):
        batch_size=100
        self.root = os.path.expanduser(root)
        self.transform = transform
        if as_train_data:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            if as_out:
                self.transform = transforms.Compose([
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.Resize((64,64)),
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                  #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                self.transform = transforms.Compose([
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        self.target_transform = target_transform
        self.db_nb = 3

        input = np.load(os.path.join(self.root, 'cifar10.npz'))
        imgs = input['CIFAR_images']
        labels = input['CIFAR_labels']
        datasize_1 = 50000
        datasize_2 = 5000
        datasize_3 = 5000
        datasize = [0,datasize_1,datasize_1+datasize_2,datasize_1+datasize_2+datasize_3]

        np.random.seed(666)
        perm = np.arange(len(imgs))
        np.random.shuffle(perm)
        imgs = imgs[perm]
        labels = labels[perm]

        tmp_imgs=imgs[datasize[0]:datasize[1]]
        tmp_labels=labels[datasize[0]:datasize[1]]
        np.random.seed(123)
        perm = np.arange(len(tmp_imgs))
        np.random.shuffle(perm)
        tmp_imgs = tmp_imgs[perm]
        tmp_labels = tmp_labels[perm]

        # for D1 to D3
        if db_idx>=0 and db_idx<self.db_nb:
            if stacking_flag == True and db_idx == 0:
                self.data = imgs[int(stacking_idx * datasize_1 / 3):int((stacking_idx+1) * datasize_1 / 3)]
                self.labels = labels[int(stacking_idx * datasize_1 / 3):int((stacking_idx + 1) * datasize_1 / 3)]
            else:
                self.data = imgs[datasize[db_idx]:datasize[db_idx+1]]
                self.labels = labels[datasize[db_idx]:datasize[db_idx+1]]
        # ----------------------------------
        # for training attack model(strong attack)
        # ----------------------------------
        elif db_idx == 3:       #for top 50% D1 as train members
            self.data = tmp_imgs[datasize[0]:int(datasize_1*0.5)]
            self.labels = tmp_labels[datasize[0]:int(datasize_1*0.5)]
        elif db_idx == 4:       #for last 50% D1 as test members
            self.data =  tmp_imgs[int(datasize_1*0.5):datasize_1]
            self.labels =  tmp_labels[int(datasize_1*0.5):datasize_1]
        elif db_idx == 5:       #for top 50% D3 as train nonmembers
            self.data = np.repeat(imgs[datasize[2]:datasize[2]+int(datasize_3*0.5)],int(datasize_1/datasize_3),axis=0)[:int(0.5*datasize_1)]
            self.labels = np.repeat(labels[datasize[2]:datasize[2]+int(datasize_3*0.5)],int(datasize_1/datasize_3),axis=0)[:int(0.5*datasize_1)]
        elif db_idx == 6:       #for last 50% D3 as test nonmembers
            self.data = np.repeat(imgs[datasize[2]+int(datasize_3*0.5):datasize[3]],int(datasize_1/datasize_3),axis=0)[:int(0.5*datasize_1)]
            self.labels = np.repeat(labels[datasize[2]+int(datasize_3*0.5):datasize[3]],int(datasize_1/datasize_3),axis=0)[:int(0.5*datasize_1)]
        # ----------------------------------
        # for training shadowmodel and corresponding attack model(weak attacker)
        # ----------------------------------
        elif db_idx == 7:       #for top 50% (x50% D1 and D3) as train members (equal to D3')
            imgs_1 = tmp_imgs[datasize[0]:int(datasize_1*0.5)]
            labels_1 = tmp_labels[datasize[0]:int(datasize_1*0.5)]
            imgs_2 = imgs[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            labels_2 = labels[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            imgs_tmp = np.concatenate([imgs_1, imgs_2], axis=0)
            labels_tmp = np.concatenate([labels_1, labels_2], axis=0)
            perm = np.arange(len(imgs_tmp))
            np.random.seed(666)
            np.random.shuffle(perm)
            self.data = imgs_tmp[perm]
            self.labels = labels_tmp[perm]
            self.data = self.data[:int(0.5*len(self.data))]
            self.labels = self.labels[:int(0.5*len(self.labels))]
        elif db_idx == 8:       #for last 50% (x50% D1 and D3) as train non members (equal to D3")
            imgs_1 = tmp_imgs[datasize[0]:int(datasize_1*0.5)]
            labels_1 = tmp_labels[datasize[0]:int(datasize_1*0.5)]
            imgs_2 = imgs[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            labels_2 = labels[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            imgs_tmp = np.concatenate([imgs_1, imgs_2], axis=0)
            labels_tmp = np.concatenate([labels_1, labels_2], axis=0)
            perm = np.arange(len(imgs_tmp))
            np.random.seed(666)
            np.random.shuffle(perm)
            self.data = imgs_tmp[perm]
            self.labels = labels_tmp[perm]
            self.data = self.data[int(0.5*len(self.data)):]
            self.labels = self.labels[int(0.5*len(self.labels)):]
        # ----------------------------------
        # for model inversion
        # ----------------------------------
        elif db_idx == 9:       #80% D1+D2+D3 data for train
            imgs_1 = imgs[datasize[0]:int(datasize_1*0.8)]
            labels_1 = labels[datasize[0]:int(datasize_1*0.8)]
            imgs_2 = imgs[datasize[1]:datasize[1]+int(datasize_2*0.8)]
            labels_2 = labels[datasize[1]:datasize[1]+int(datasize_2*0.8)]
            imgs_3 = imgs[datasize[2]:datasize[2]+int(datasize_3*0.8)]
            labels_3 = labels[datasize[2]:datasize[2]+int(datasize_3*0.8)]
            self.data = np.concatenate([imgs_1, imgs_2, imgs_3], axis=0)
            self.labels = np.concatenate([labels_1, labels_2, labels_3], axis=0)
        elif db_idx == 10:      #20% D1 data for test1
            self.data = imgs[int(datasize_1*0.8):datasize[1]]
            self.labels = labels[int(datasize_1*0.8):datasize[1]]
        elif db_idx == 11:      #20% D2+D3 data for test2
            imgs_2 = imgs[datasize[1]+int(datasize_2*0.8):datasize[2]]
            labels_2 = labels[datasize[1]+int(datasize_2*0.8):datasize[2]]
            imgs_3 = imgs[datasize[2]+int(datasize_3*0.8):datasize[3]]
            labels_3 = labels[datasize[2]+int(datasize_3*0.8):datasize[3]]
            self.data = np.concatenate([imgs_2, imgs_3], axis=0)
            self.labels = np.concatenate([labels_2, labels_3], axis=0)
        else:
            raise Exception('Error! The database index {} exceeds total databases amount {}'.format(db_idx,db_nb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

if __name__=='__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    a=LOAD_DATASET('../data', transform=transform, db_idx=2,index=True)
