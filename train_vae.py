
from utils import *
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import time
import importlib
from model.vae import VAE
import config.config_common as config_common
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='Membership Inference Attack Demo', conflict_handler='resolve')
parser.add_argument('--batch-size', type=int, default=50, metavar='')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument("--dataset", type=str,required=True,default='purchase', help="name of datasets")
parser.add_argument('--trainshadowmodel',action='store_true',default=False)
parser.add_argument('--setup',type=str,required=False, default='base',help="means the output path of purifier")
BCE_loss = nn.BCELoss(reduction = "sum")
def loss(X, X_hat, mean, logvar):
    reconstruction_loss = BCE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #missing KLD
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_purifier_with_helper(opt, purifier, batchsize, log_interval, device,
                               data_loader, optimizer, epoch):
    criterion = nn.BCELoss()
    # generate the label of F(x)
    real_label = torch.full((batchsize,), 1, device=device, requires_grad=False).float()
    fake_label = torch.full((batchsize,), 0, device=device, requires_grad=False).float()
    for batch_idx, (data, target, raw) in enumerate(data_loader):
        # data: classifier's softlabel F(x)
        # target: ground-truth label of x
        # raw: input data x
        data, target, raw = data.to(device).float(), target.to(device).float(), raw.to(device).float()

        # Compute one_hot encoding of target
        target_oh = torch.from_numpy((np.zeros((target.size(0), opt.featuresize)))).to(device).float()
        target_oh = target_oh.scatter_(1, target.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)

        # here data_softmax is softmax version data
        #data_softmax = F.softmax(data, dim=1)
        data_softmax = data

        # prediction one hot version
        pred = data_softmax.max(1, keepdim=True)[1].squeeze()

        ## Train with F(x)

        ## Train with G(F(x))
        # calculate G(F(x))
        prob_out_softmax = purifier(data_softmax, target, release='softmax')

        # -----------------
        # train the purifier
        # -----------------

        purifier.train()
        optimizer.zero_grad()

        prob_out, mu, log_var = purifier(data_softmax, pred, release='raw')

        # get the output of purifier
        prob_out_softmax = F.softmax(prob_out, dim=1)
        prob_out_logsoftmax = F.log_softmax(prob_out, dim=1)

        # calculate MSE loss of purifier
        # calculate the negative log mse loss of purifier
        prob_out_logsoftmax_neg=-prob_out_logsoftmax
        data_logsoftmax_neg=-torch.log(data_softmax)

        # errG_1 = F.mse_loss(prob_out_softmax, data_softmax)
        
        # mse_loss_each = F.mse_loss(output_softmax, data_softmax, reduction='none')

        errG_1 = F.mse_loss(prob_out_logsoftmax_neg, data_logsoftmax_neg)
        # errG_1 = -torch.log(mse_loss_each).mean()

        errG_3 = F.mse_loss(prob_out_softmax, data_softmax) 
        # # print(prob_out)
        # # print(prob_out.shape)
        # # print(data.shape)
        # # print(data)
        # errG_3 = F.nll_loss(prob_out, data)

        # prediction one hot version
        pred = data_softmax.max(1, keepdim=True)[1].squeeze()

        # errC = F.nll_loss(prob_out_logsoftmax, target.long())  # ground truth label_crossentropy
        # calculate crossentropy based on one hot prediction
        errC = F.nll_loss(prob_out_logsoftmax, pred)

        # total loss of purifier :minimize R(F(x), G(F(x))) + reg2 * crossentropy(G(F(x)),onehot(F(x))) + reg3 * BCELoss(D(G(F(x)),1)
        # if withhelper, objective function of purifier: minimize R(F(x), G(F(x))) - reg * R(x, H(G(F(x))))
        # R is the reconstruction loss
        loss = loss_function(prob_out_softmax, data_softmax, mu, log_var) # errC

        # update gradients
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\nLoss_G: {:.6f}\nLoss_G: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset), errG_3.item(), errG_1.item()))

def test_purifier_with_helper(opt, purifier, device, data_loader, epoch, dir, msg):
    purifier.eval()
    criterion = nn.BCELoss(reduction='sum')
    test_loss = 0
    mse_loss = 0
    errC = 0
    errG_2 = 0
    correct = 0
    entropy_1 = 0
    entropy_2 = 0
    confidence_score_distortion = 0
    labelloss = 0
    with torch.no_grad():
        for data, target, raw in data_loader:
            # data: classifier's softlabel F(x)
            # target: ground-truth label of x
            # raw: input data x
            data, target, raw = data.to(device).float(), target.to(device).long(), raw.to(device).float()

            # generate the label of F(x)
            real_label = torch.full((len(data),), 1, device=device, requires_grad=False).float()
            fake_label = torch.full((len(data),), 0, device=device, requires_grad=False).float()

            # Compute one_hot encoding of target
            target_oh = torch.from_numpy((np.zeros((target.size(0), opt.featuresize)))).to(device).float()
            target_oh = target_oh.scatter_(1, target.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)

            #data_softmax = F.softmax(data, dim=1)
            # here data_softmax is softmax version data
            data_softmax = data

            # prediction one hot version
            pred = data_softmax.max(1, keepdim=True)[1].squeeze()
            
            cl_pred = data_softmax.max(1, keepdim=True)[1].squeeze()

            # get the output of purifier
            output, mu, log_var = purifier(data_softmax, pred, release='log_softmax')
            output_softmax, mu, log_var = purifier(data_softmax, pred, release='softmax')

            # calculate cross entropy loss as classification loss
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()

            # caluculate MSE loss of purifier
            prob_out_logsoftmax_neg=-output
            data_logsoftmax_neg=-torch.log(data_softmax)
            # mse_loss += F.mse_loss(prob_out_logsoftmax_neg, data_logsoftmax_neg, reduction='sum').item()
            mse_loss += F.mse_loss(output_softmax, data_softmax, reduction='sum').item()

            # get prediction of purifier output
            pred = output.max(1, keepdim=True)[1]

            # calculate acc
            correct += pred.eq(target.view_as(pred)).sum().item()

            # get the prediction of classifier output
            pred_raw = data_softmax.max(1, keepdim=True)[1]

            # calculate label loss
            labelloss += len(pred) - pred.eq(pred_raw.view_as(pred)).sum().item()

            # calculate entropy before&after purifier
            entropy_1 += Categorical(probs=data_softmax).entropy()
            entropy_2 += Categorical(probs=output_softmax).entropy()

            # calculate confidence score distortion after purifier
            confidence_score_distortion += (output_softmax - data_softmax).abs().sum()

    # print progress and calculate means of metrics
    test_loss /= (len(data_loader.dataset) * opt.featuresize)
    mse_loss /= (len(data_loader.dataset) * opt.featuresize)
    # mse_loss /= (len(data_loader.dataset)) * 3
    #errH_2 /= len(data_loader.dataset) * np.cumprod(data.shape[1:])[-1]
    entropy_1 = (entropy_1.sum() / len(data_loader.dataset)) / np.log(opt.featuresize)
    entropy_2 = (entropy_2.sum() / len(data_loader.dataset)) / np.log(opt.featuresize)
    confidence_score_distortion /= len(data_loader.dataset)
    labelloss /= len(data_loader.dataset)

    print('Test classifier on {} set:\nAverage loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\nMSELoss: {:.6f}\nentropy_before: {:.6f}'
          '\nentropy_after: {:.6f}\nlabelloss: {:.6f}%\nconfidence_score_distortion: {:.6f}\n'.format(
            msg, test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset), mse_loss,
            entropy_1, entropy_2, 100. * labelloss, confidence_score_distortion))
    return (correct / len(data_loader.dataset), test_loss), \
           (mse_loss, mse_loss), \
           (labelloss, confidence_score_distortion.item(), entropy_1.item(), entropy_2.item())


def trainPurifierModel(opt,size,training_idx,repeat):
    # GPU setting
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': opt.num_workers, 'pin_memory': True} if use_cuda else {}
    #print(torch.cuda.device_count())
    basedir = config_common.DIR_TO_SAVE_TRAINED_VAES_AND_RESULTS+'/'
    os.makedirs(basedir,exist_ok=True)
    dataset = importlib.import_module('dataset.{}'.format(opt.dataset))
    net = importlib.import_module('model.{}'.format(opt.dataset))

    # load dataset for training purifier model
    # db_idx=1 (stands for D2) that is used to train the purifier
    # db_idx=0&2 (D1&``D3``) are test datasets on target model
    # db_idx = 7(stands for top50 % of shuffled(top50 % D1 + top50 % D3)) as training data
    # and db_idx=8 (stands for lats50% of shuffled (top50% D1 + top50% D3)) as test data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA, transform=transform,
                                         db_idx=training_idx)
    if len(dataset_train) != size:
        raise Exception('Invalid config!')
    
    # dataset_train = dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA, transform=transform, db_idx=8, index=True, group=2)
    # dataset_test1 = dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA, transform=transform, db_idx=8, index=True, group=1)
    # dataset_test2 = dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA, transform=transform, db_idx=8, index=True, group=3)
    dataset_test1 = dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA, transform=transform,
                                        db_idx=7) if opt.trainshadowmodel else dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA,
                                                                                                    transform=transform,
                                                                                                    db_idx=0)
    dataset_test2 = dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA, transform=transform,
                                         db_idx=8) if opt.trainshadowmodel else dataset.LOAD_DATASET(config_common.DIR_TO_SAVE_DATA,
                                                                                                     transform=transform,
                                                                                                     db_idx=2)

    # create dataloader for classifier evaling
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False, **kwargs)
    dataloader_test1 = torch.utils.data.DataLoader(dataset_test1, batch_size=opt.batch_size,
                                                                 shuffle=False, **kwargs)
    dataloader_test2 = torch.utils.data.DataLoader(dataset_test2, batch_size=opt.batch_size, shuffle=False, **kwargs)

    # determine the classifiers used later
    classifier = nn.DataParallel(net.Classifier()).to(device)
    # For cifar10, hidden_sizes = None
    # For purchase, hidden_sizes = [128, 256, 512]
    # For facescrub, hidden_sizes = [512, 1024, 2048]

    if opt.dataset in ['purchase','texas','location']:
        hidden_sizes = [128, 256, 512]
        latent_size = 20
    elif opt.dataset in ['facescrub', 'cifar100']:
        hidden_sizes = [512, 1024, 2048]
        latent_size = 100
    else:
        hidden_sizes = [32, 64, 128]
        latent_size = 2

    # For cifar10, latent_size = 2
    # For purchase, latent_size = 20
    # For facescrub, latent_size = 100
    # purifier = nn.DataParallel(VAE(input_size=opt.featuresize, latent_size=latent_size, hidden_sizes= hidden_sizes)).to(device)
    purifier = VAE(input_size=opt.featuresize, latent_size=latent_size, hidden_sizes= hidden_sizes).to(device)

    # Apply the weights_init function to randomly initialize all weights
    purifier.apply(weights_init_normal)
    
    # determine the optimizer of each model
    optimizer = optim.Adam(purifier.parameters(), lr=opt.lr, betas=(0.5, 0.999), amsgrad=True)

    # check the model
    print('vae:\n', purifier)

    # Load classifier
    if opt.trainshadowmodel:
        path = os.path.join(opt.dataset, opt.dataset + '_shadowmodel.pth')
    else:
        path = os.path.join(config_common.DIR_TO_SAVE_TRAINED_CLASSIFIERS_AND_RESULTS, opt.dataset + '_targetmodel.pth')
    # Load method (origin or not)
    classifier = net.load_classifier(classifier, path)

    # read the confidence score of classifier
    softlabel_train = classification(classifier, purifier, device, dataloader_train)
    softlabel_test1 = classification(classifier, purifier, device, dataloader_test1)
    softlabel_test2 = classification(classifier, purifier, device, dataloader_test2)

    # if opt.as_attack == True:
    #     softlabel_train = np.load('facescrub/facescrub_softlabel_memguard_D1_1.0.npz', allow_pickle=True)['data']
    #     softlabel_test1 = np.load('facescrub/facescrub_softlabel_memguard_D0_1.0.npz', allow_pickle=True)['data']
    #     softlabel_test2 = np.load('facescrub/facescrub_softlabel_memguard_D1_1.0.npz', allow_pickle=True)['data'][:8000]

    # create dataset for training purifier&helper
    if (opt.dataset == 'cifar10' or opt.dataset == 'cifar100' or opt.dataset == 'facescrub' or opt.dataset == 'mnist'):
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = None

    # transfroms for the raw imgs for calculation of  reconstruction loss
    dataset_train = PURIFIER_DATASET(softlabel_train, dataset_train.labels, dataset_train.data, transform=transform)
    dataset_test1 = PURIFIER_DATASET(softlabel_test1, dataset_test1.labels, dataset_test1.data, transform=transform)
    dataset_test2 = PURIFIER_DATASET(softlabel_test2, dataset_test2.labels, dataset_test2.data, transform=transform)

    # create the dataloader for training purifier&helper
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, **kwargs)
    dataloader_train_as_evaluation = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,
                                                                 shuffle=False, **kwargs)
    dataloader_test_1 = torch.utils.data.DataLoader(dataset_test1, batch_size=opt.batch_size,
                                                    shuffle=False, **kwargs)
    dataloader_test_2 = torch.utils.data.DataLoader(dataset_test2, batch_size=opt.batch_size,
                                                    shuffle=False, **kwargs)

    # logs
    train_purifier = []
    train_classifier = []
    train_discriminator = []
    train_metrics = []

    test1_purifier = []
    test1_classifier = []
    test1_discriminator = []
    test1_metrics = []

    test2_purifier = []
    test2_classifier = []
    test2_discriminator = []
    test2_metrics = []

    time_list = []

    best_loss = 999
    best_acc = 0
    min_label_loss=1
    best_epoch = 0

    # Determine the model name and metrics name according to the arguments
    if opt.trainshadowmodel:
        purifiermodel_name = basedir+opt.dataset  +'_shadow_ri_cvaemodel.pth'
        dataname = basedir +opt.dataset+ '_shadow_ri_cvae_metrics.npz'
    else:
        purifiermodel_name = basedir+opt.dataset  +'_target_ri_cvaemodel.pth'
        dataname = basedir+opt.dataset + '_target_ri_cvae_metrics.npz'

    for epoch in range(1, opt.epochs + 1):
        print('#-----------------------------------------')
        # print and save timestamps
        print(time.time())
        start = time.time()
        # train the purifier, helper and discriminator
        train_purifier_with_helper(opt, purifier, opt.batch_size, opt.log_interval,
                                   device, dataloader_train, optimizer, epoch)
        time_list.append(time.time() - start)
        print(time.time())

        # read acc and loss of training data
        print('D2 results:')
        cl_2, errG, metrics_2 = test_purifier_with_helper( opt, purifier, device,
                                                             dataloader_train_as_evaluation, epoch, '../figs', 'train')
        train_purifier.append(errG)
        train_classifier.append(cl_2)
        train_metrics.append(metrics_2)

        # read acc and loss of test1 data
        print('D1 results:')
        cl, errG, metrics = test_purifier_with_helper(opt, purifier, device,
                                                            dataloader_test_1, epoch, '../figs', 'test1')
        test1_purifier.append(errG)
        test1_classifier.append(cl)
        test1_metrics.append(metrics)

        # read acc and loss of test2 data
        print('D3 results:')
        cl, errG, metrics = test_purifier_with_helper(opt, purifier, device,
                                                            dataloader_test_2, epoch, '../figs', 'test2')
        test2_purifier.append(errG)
        test2_classifier.append(cl)
        test2_metrics.append(metrics)

        # if min_label_loss >= metrics_2[0]:
        if best_loss >= errG[0]:
            # save best purifier model based on D3 acc
            print('saving vae model')
            best_loss = errG[0]
            best_acc = cl_2[0]
            best_epoch = epoch
            min_label_loss=metrics_2[0]
            state_purifier = {
                'epoch': epoch,
                'model': purifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': errG[0]
            }
            torch.save(state_purifier, purifiermodel_name)

    print("Best vae: epoch {}, acc {:.4f}".format(best_epoch, best_acc))
    # save metrics
    np.savez(dataname,
             train_purifier=np.array(train_purifier),
             train_classifier=np.array(train_classifier),
             train_metrics=np.array(train_metrics),
             test1_purifier=np.array(test1_purifier),
             test1_classifier=np.array(test1_classifier),
             test1_metrics=np.array(test1_metrics),
             test2_purifier=np.array(test2_purifier),
             test2_classifier=np.array(test2_classifier),
             test2_metrics=np.array(test2_metrics),
             )
    return

def main():
    args = parser.parse_args()

    f = open(config_common._PROJECT_PATH+"/config/purifier.json", encoding="utf-8")
    content = json.loads(f.read())

    parser.add_argument('--epochs', type=int, default=content[args.dataset]['epochs'], metavar='')
    parser.add_argument('--lr', type=float, default=content[args.dataset]['lr'], metavar='')
    # parser.add_argument('--lrH', type=float, default=content[args.dataset]['lrH'], metavar='')
    # parser.add_argument('--lrD', type=float, default=content[args.dataset]['lrD'], metavar='')
    parser.add_argument('--featuresize', type=int, default=content[args.dataset]['featuresize'], metavar='')

    f = open(config_common._PROJECT_PATH+"/config/vae.json", encoding="utf-8")
    content = json.loads(f.read())
    
    parser.add_argument('--size',type=int, default=content[args.dataset][args.setup]["size"])
    parser.add_argument('--idx',type=int, default=content[args.dataset][args.setup]["dataset_index"])
    parser.add_argument('--repeat', action='store_true', default=content[args.dataset][args.setup]["repeat"])

    args = parser.parse_args()

    print("Training vae Model!")
    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.dataset, exist_ok=True)

    
    # confirm the arguments
    print("================================")
    print(args)
    print("================================")
    # size = [5000,10000,20000,40000,60000,20000]
    # idx = [1,1,1,1,1,0]
    # repeat = [False,False,False,True,True,False]
    # ----------
    #  train purifier model
    # ----------
    trainPurifierModel(opt=args,size=args.size,training_idx=args.idx,repeat=args.repeat)

if __name__ == '__main__':
    main()