import importlib, os, sys
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from utils import *
import config.config_common as config_common
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    root = "Path/To/Purifier"
    dataset_str = 'cifar10' 
    path_classifier = root+'trained_classifiers_and_results/'+dataset_str+'_targetmodel.pth'

    ########################################## Instantiate and Initialize Classifier ##########################################
    module_net = importlib.import_module(f'model.{dataset_str}')     
    # classifier = module_net.Classifier().to('cuda:0')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    classifier = nn.DataParallel(module_net.Classifier()).to(device)
    classifier = load_classifier(classifier, path_classifier)
    
    for i in range(0, 9):
        ############################### Get Dataloader ###############################
        module_dataset = importlib.import_module(f'dataset.{dataset_str}') 
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = module_dataset.LOAD_DATASET(root+'data',transform=transform,db_idx=i)
        # dataset = module_dataset.LOAD_DATASET(db_idx=i, mode="valid") 
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False) 

        ########################## Calculate and Save Softmax ##########################
        _DIR = os.path.join(config_common.DIR_TO_SAVE_SOFTMAX_CLASSIFIER, f'{dataset_str}_targetmodel')
        if not os.path.exists(_DIR):
            os.makedirs(_DIR)

        _NAME = f'{dataset_str}_targetmodel_D{i}.pth'
        _PATH = os.path.join(_DIR, _NAME)
        softmax, acc = get_save_or_load_softmax(classifier, dataloader, device, dataset_str, _PATH)
        print()
