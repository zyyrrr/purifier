from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class LOAD_DATASET(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_idx=0, stacking_flag=False,stacking_idx=0):

        self.root = os.path.expanduser(root)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.target_transform = target_transform
        self.db_nb = 3

        input = np.load(os.path.join(self.root, 'facescrub.npz'))
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']

        imgs = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)

        datasize_1 = 30000
        datasize_2 = 10000
        datasize_3 = 8000
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
            self.data = tmp_imgs[int(datasize_1*0.5):datasize_1]
            self.labels = tmp_labels[int(datasize_1*0.5):datasize_1]
        elif db_idx == 5:       #for top 50% D3 as train nonmembers
            self.data = np.repeat(imgs[datasize[2]:datasize[2]+int(datasize_3*0.5)],4,axis=0)[:int(0.5*datasize_1)]
            self.labels = np.repeat(labels[datasize[2]:datasize[2]+int(datasize_3*0.5)],4,axis=0)[:int(0.5*datasize_1)]
        elif db_idx == 6:       #for last 50% D3 as test nonmembers
            self.data = np.repeat(imgs[datasize[2]+int(datasize_3*0.5):datasize[3]],4,axis=0)[:int(0.5*datasize_1)]
            self.labels = np.repeat(labels[datasize[2]+int(datasize_3*0.5):datasize[3]],4,axis=0)[:int(0.5*datasize_1)]
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
        # test1 for facescrub is D1 (db_idx=0)
        # training data for facescrub is CelebA
        elif db_idx == 9:     #test2 data for test (whole D2+D3)
            imgs_2 = imgs[datasize[1]:datasize[2]]
            labels_2 = labels[datasize[1]:datasize[2]]
            imgs_3 = imgs[datasize[2]:datasize[3]]
            labels_3 = labels[datasize[2]:datasize[3]]
            self.data = np.concatenate([imgs_2, imgs_3], axis=0)
            self.labels = np.concatenate([labels_2, labels_3], axis=0)
        elif db_idx == 10:
            self.data = imgs[:datasize[-1]]
            self.labels = labels[:datasize[-1]]
        # ----------------------------------
        # for Attacks that requires different D2
        # ----------------------------------
        # test1 for facescrub is D1 (db_idx=0)
        # training data for facescrub is CelebA  
        elif db_idx == 11:       #for top 50% D2 as train members
            self.data = np.repeat(imgs[datasize[1]:datasize[1]+int(datasize_2*0.5)],6,axis=0)[:int(0.5*datasize_1)]
            self.labels = np.repeat(labels[datasize[1]:datasize[1]+int(datasize_2*0.5)],6,axis=0)[:int(0.5*datasize_1)]
        elif db_idx == 12:       #for last 50% D2 as test members
            self.data = np.repeat(imgs[datasize[1]+int(datasize_2*0.5):datasize[2]],6,axis=0)[:int(0.5*datasize_1)]
            self.labels = np.repeat(labels[datasize[1]+int(datasize_2*0.5):datasize[2]],6,axis=0)[:int(0.5*datasize_1)]
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

class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None, mode='all', size=64):
        self.root = os.path.expanduser(root)
        # self.root = root
        self.transform = transform
        self.target_transform = target_transform

        data = []
        for i in range(10):
            data.append(np.load('./data/celebA_64_{}.npy'.format(i + 1)))
            # data.append(np.load(root+'/data/celebA_64_{}.npy'.format(i + 1)))
        data = np.concatenate(data, axis=0)

        # v_min = data.min(axis=0)
        # v_max = data.max(axis=0)
        # data = (data - v_min) / (v_max - v_min)
        labels = np.array([0] * len(data))

        if mode == 'train':
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        if mode == 'test':
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]
        if mode == 'all':
            self.data = data
            self.labels = labels
        if mode == 'quarter':
            self.data = data[:int(0.25 * len(data))]
            self.labels = labels[:int(0.25 * len(data))]

        print('data:', self.data.shape, self.data.min(), self.data.max())
        print('labels:', self.labels.shape, len(np.unique(self.labels)), 'unique labels')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target