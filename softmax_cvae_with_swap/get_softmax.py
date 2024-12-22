from torch.utils.data.dataloader import DataLoader
from utils import *
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch; torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import config.config_common as config_common
from config import config_train_vae
from torchvision import transforms
from model.vae import *
import importlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
# from utils import SOFTLABEL_WITH_CLASS, SOFTLABEL_DATASET
from label_swapper import label_swapper_dynamic
from tqdm import tqdm
# from config import config_train_vae

torch.set_num_threads(40)

def _get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Membership Inference Attack Demo', conflict_handler='resolve')


    parser.add_argument('--batch_size', type=int, default=1024, metavar='')
    parser.add_argument('--test_batch_size', type=int, default=1024, metavar='')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='')
    parser.add_argument('--log_interval', type=int, default=10, metavar='')
    parser.add_argument('--num_workers', type=int, default=1, metavar='')
    parser.add_argument("--dataset", type=str, help="name of datasets")
    parser.add_argument("--trainshadowmodel", action='store_true', default=False, help="train a shadow model, if false then train a target model")
    parser.add_argument('--use_purifier',action='store_true', default=True)

    args = parser.parse_args()

    # parser.add_argument('--epochs', type=int, default=config_train_vae.CONFIG[args.dataset]['epochs'], metavar='')
    # parser.add_argument('--epochs_classifier', type=int, default=config_train_vae.CONFIG[args.dataset]['epochs'], metavar='')
    parser.add_argument('--featuresize', type=int, default=config_train_vae.CONFIG[args.dataset]['featuresize'], metavar='')

    print("======================= args =======================")
    print(args)

    return parser.parse_args()

def process(args, label_swapper, cvae, device, index):

    ###################################### Dataloader ######################################
    module_dataset = importlib.import_module('dataset.{}'.format(args.dataset))  # module: dataset.{dataset}
    transform = transforms.Compose([transforms.ToTensor()])                 
    dataset = module_dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA, transform=transform, db_idx=index)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ################################### Softmax of Swapper ###################################
    _DIR = os.path.join(config_common.DIR_TO_SAVE_SOFTMAX_SWAPPER, f'swapper_targetmodel_{args.dataset}')
    if not os.path.exists(_DIR):
        os.makedirs(_DIR)

    _NAME_SOFTMAX_SWAPPER= f'swapper_targetmodel_{args.dataset}_D{index}.pth'
    _PATH_TO_SAVE_SOFTMAX_SWAPPER = os.path.join(_DIR, _NAME_SOFTMAX_SWAPPER)

    softmax_swapper, _, label_ori = get_save_or_load_softmax(label_swapper, dataloader, device, args.dataset, _PATH_TO_SAVE_SOFTMAX_SWAPPER, return_label_ori=True)
    softmax_swapper = softmax_swapper.to(device)
    
    softmax_tmp = torch.load(config_common.DIR_TO_SAVE_SOFTMAX_CLASSIFIER+f"/{args.dataset}_targetmodel/{args.dataset}_targetmodel_D{index}.pth")['softmax'].to(device)
    
    print(f'num of softlabels_swapped - softmax_tmp: {((softmax_tmp != softmax_swapper).sum(dim=1).sum(dim=0))/2}')

    ############################## Softmax of CVAE after Swapper ##############################
    softmax_cvae_with_swapper, mius, logvars = cvae(softmax_swapper, softmax_swapper.argmax(dim=-1))
    print(mius)
    print(cvae(softmax_tmp, softmax_tmp.argmax(dim=-1))[1])
    
    _DIR = os.path.join(config_common.DIR_TO_SAVE_SOFTMAX_CVAE_WITH_SWAP, f'cvae_targetmodel_{args.dataset}')
    if not os.path.exists(_DIR):
        os.makedirs(_DIR)

    _NAME_SOFTMAX_CVAE_WITH_SWAPPER = f'cvae_targetmodel_{args.dataset}_D{index}.pth'
    _PATH_TO_SAVE_SOFTMAX_CVAE_WITH_SWAPPER = os.path.join(_DIR, _NAME_SOFTMAX_CVAE_WITH_SWAPPER)
    torch.save(
        {
            'label_ori': label_ori.cpu(),
            'softmax': softmax_cvae_with_swapper.cpu(),
            'hiddins_results': softmax_cvae_with_swapper.cpu()
        },
        _PATH_TO_SAVE_SOFTMAX_CVAE_WITH_SWAPPER
    )

    _NAME_MIUS_LOGVARS = f'mius_logvars_{args.dataset}_D{index}.pth'
    _SUB_DIR = os.path.join(config_common.DIR_TO_SAVE_MIUS_AND_LOGVARS_WITH_SWAP, f'targetmodel_{args.dataset}')
    if not os.path.exists(_SUB_DIR):
        os.makedirs(_SUB_DIR, exist_ok=True)
    _PATH_MIUS_LOGVARS = os.path.join(_SUB_DIR, _NAME_MIUS_LOGVARS)
    torch.save({
        'mius': mius.cpu(),
        'logvars': logvars.cpu(),
        'label_ori': label_ori.cpu()
    }, _PATH_MIUS_LOGVARS)

def main():
    args = _get_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ##################################### Instantiate and Initialize Swapper #####################################
    label_swapper = label_swapper_dynamic(args.dataset)

    ####################################### Instantiate and Initialize CVAE #######################################
    _NAME_CVAE = f'{args.dataset}_target_ri_cvaemodel.pth'
    _PATH_CVAE = os.path.join(config_common.DIR_TO_SAVE_TRAINED_VAES_AND_RESULTS, _NAME_CVAE)
    # cvae = VAE(args.dataset).to(device)
    if args.dataset in ['purchase','texas','location']:
        hidden_sizes = [128, 256, 512]
        latent_size = 20
    elif args.dataset in ['facescrub', 'cifar100']:
        hidden_sizes = [512, 1024, 2048]
        latent_size = 100
    # elif args.dataset in ['cifar10', 'UTKFace']:
    elif args.dataset in ['cifar10']:
        hidden_sizes = [32, 64, 128]
        latent_size = 2
    # elif args.dataset in ['UTKFace']:
    #     hidden_sizes = [128, 256, 512]
    #     latent_size = 64
    cvae = VAE(input_size=args.featuresize, latent_size=latent_size, hidden_sizes= hidden_sizes).to(device)
    load_vae(cvae, _PATH_CVAE)
    cvae.eval()

    for i in range(0, 9):
        process(args, label_swapper, cvae, device, index=i)

if __name__ == '__main__':
    main()

