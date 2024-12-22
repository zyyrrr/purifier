from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pdb

class LOAD_DATASET(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_idx=0, D2_size=20000, stacking_flag=False,stacking_idx=0):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.db_nb = 3

        input = np.load(os.path.join(self.root, 'purchase.npz'))
        data = input['data']
        labels = input['label']

        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]

        datasize_1 = 20000
        datasize_2 = D2_size
        datasize_3 = 20000
        datasize = [0,datasize_1,datasize_1+datasize_2,datasize_1+datasize_2+datasize_3]

        tmp_data=data[datasize[0]:datasize[1]]
        tmp_labels=labels[datasize[0]:datasize[1]]
        np.random.seed(123)
        perm = np.arange(len(tmp_data))
        np.random.shuffle(perm)
        tmp_data = tmp_data[perm]
        tmp_labels = tmp_labels[perm]

        if db_idx>=0 and db_idx<self.db_nb:       #for D1 to D3
            if stacking_flag == True and db_idx == 0:
                self.data = data[int(stacking_idx * datasize_1 / 3):int((stacking_idx+1) * datasize_1 / 3)]
                self.labels = labels[int(stacking_idx * datasize_1 / 3):int((stacking_idx + 1) * datasize_1 / 3)]
            else:    
                self.data = data[datasize[db_idx]:datasize[db_idx+1]]
                self.labels = labels[datasize[db_idx]:datasize[db_idx+1]]
        # ----------------------------------
        # for training attack model(strong attack)
        # ----------------------------------
        elif db_idx == 3:       #for top 50% D1 as train members
            self.data = tmp_data[datasize[0]:int(datasize_1*0.5)]
            self.labels = tmp_labels[datasize[0]:int(datasize_1*0.5)]
        elif db_idx == 4:       #for last 50% D1 as test members
            self.data = tmp_data[int(datasize_1*0.5):datasize_1]
            self.labels = tmp_labels[int(datasize_1*0.5):datasize_1]
        elif db_idx == 5:       #for top 50% D3 as train nonmembers
            self.data = np.repeat(data[datasize[2]:datasize[2]+int(datasize_3*0.5)],int(datasize_1/datasize_3),axis=0)
            self.labels = np.repeat(labels[datasize[2]:datasize[2]+int(datasize_3*0.5)],int(datasize_1/datasize_3),axis=0)
        elif db_idx == 6:       #for last 50% D3 as test nonmembers
            self.data = np.repeat(data[datasize[2]+int(datasize_3*0.5):datasize[3]],int(datasize_1/datasize_3),axis=0)
            self.labels = np.repeat(labels[datasize[2]+int(datasize_3*0.5):datasize[3]],int(datasize_1/datasize_3),axis=0)
        # ----------------------------------
        # for training shadowmodel and corresponding attack model(weak attacker)
        # ----------------------------------
        elif db_idx == 7:       #for top 50% (x50% D1 and D3) as train members (equal to D3')
            data_1 = tmp_data[datasize[0]:int(datasize_1*0.5)]
            labels_1 = tmp_labels[datasize[0]:int(datasize_1*0.5)]
            data_2 = data[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            labels_2 = labels[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            data_tmp = np.concatenate([data_1, data_2], axis=0)
            labels_tmp = np.concatenate([labels_1, labels_2], axis=0)
            perm = np.arange(len(data_tmp))
            np.random.seed(666)
            np.random.shuffle(perm)
            self.data = data_tmp[perm]
            self.labels = labels_tmp[perm]
            self.data = self.data[:int(0.5*len(self.data))]
            self.labels = self.labels[:int(0.5*len(self.labels))]
        elif db_idx == 8:       #for last 50% (x50% D1 and D3) as train non members (equal to D3")
            data_1 = tmp_data[datasize[0]:int(datasize_1*0.5)]
            labels_1 = tmp_labels[datasize[0]:int(datasize_1*0.5)]
            data_2 = data[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            labels_2 = labels[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            data_tmp = np.concatenate([data_1, data_2], axis=0)
            labels_tmp = np.concatenate([labels_1, labels_2], axis=0)
            perm = np.arange(len(data_tmp))
            np.random.seed(666)
            np.random.shuffle(perm)
            self.data = data_tmp[perm]
            self.labels = labels_tmp[perm]
            self.data = self.data[int(0.5*len(self.data)):]
            self.labels = self.labels[int(0.5*len(self.labels)):]
        # ----------------------------------
        # for model inversion
        # ----------------------------------
        elif db_idx == 9:       #80% D1+D2+D3 data for train
            data_1 = data[datasize[0]:int(datasize_1*0.8)]
            labels_1 = labels[datasize[0]:int(datasize_1*0.8)]
            data_2 = data[datasize[1]:datasize[1]+int(datasize_2*0.8)]
            labels_2 = labels[datasize[1]:datasize[1]+int(datasize_2*0.8)]
            data_3 = data[datasize[2]:datasize[2]+int(datasize_3*0.8)]
            labels_3 = labels[datasize[2]:datasize[2]+int(datasize_3*0.8)]
            self.data = np.concatenate([data_1, data_2, data_3], axis=0)
            self.labels = np.concatenate([labels_1, labels_2, labels_3], axis=0)
        elif db_idx == 10:      #20% D1 data for test1
            self.data = data[int(datasize_1*0.8):datasize[1]]
            self.labels = labels[int(datasize_1*0.8):datasize[1]]
        elif db_idx == 11:      #20% D2&D3 data for test2
            data_2 = data[datasize[1]+int(datasize_2*0.8):datasize[2]]
            labels_2 = labels[datasize[1]+int(datasize_2*0.8):datasize[2]]
            data_3 = data[datasize[2]+int(datasize_3*0.8):datasize[3]]
            labels_3 = labels[datasize[2]+int(datasize_3*0.8):datasize[3]]
            self.data = np.concatenate([data_2, data_3], axis=0)
            self.labels = np.concatenate([labels_2, labels_3], axis=0)
        else:
            raise Exception('Error! The database index {} exceeds total databases amount {}'.format(db_idx,db_nb))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]
        '''
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            labels = self.target_transform(labels)
        '''

        return data, labels