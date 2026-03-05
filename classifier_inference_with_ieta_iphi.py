import sys, yaml, h5py, random
import gc
import numpy as np
import os, glob
import time
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from dataset_loder import *
import pandas as pd
import pickle
import wandb
import importlib
from sklearn.metrics import roc_curve, auc

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if len(sys.argv) != 2:
    print("Usage: python XXX.py <config_file.yaml>")
    sys.exit(1)

config_file = sys.argv[1]

try:
    config = read_config(config_file)
    hyper_parameters = config.get('hyper_parameters', {})
    model_info = config.get('model_info', {})
    wandb_parameters = config.get('wandb_parameters', {})
    dir_info = config.get('dir_info', {})
    scales = config.get('scales', {})


    load_epoch= hyper_parameters.get('load_epoch')
    epochs= hyper_parameters.get('epochs')
    lr_init= hyper_parameters.get('lr_init')
    lr_factor= hyper_parameters.get('lr_factor')
    new_lr= hyper_parameters.get('new_lr')
    reslayers= hyper_parameters.get('reslayers')
    resblocks= hyper_parameters.get('resblocks')
    channels= hyper_parameters.get('channels')
    loss_func= hyper_parameters.get('loss_func')
    scheduler_= hyper_parameters.get('scheduler')
    optimizer_= hyper_parameters.get('optimizer')
    patience= hyper_parameters.get('patience')
    scheduler_mode= hyper_parameters.get('scheduler_mode')
    BATCH_SIZE= hyper_parameters.get('batch_size')
    VAL_BATCH_SIZE= hyper_parameters.get('valid_batch_size')
    TEST_BATCH_SIZE= hyper_parameters.get('test_batch_size')
    indices= hyper_parameters.get('channels')
    n_train= hyper_parameters.get('n_train')
    n_valid= hyper_parameters.get('n_valid')
    n_test= hyper_parameters.get('n_test')


    m0_scale= scales.get('m0_scale')



    run_print=model_info.get('run_print')
    cuda=model_info.get('cuda')
    timestr=model_info.get('timestr')
    random_seed=model_info.get('random_seed')
    model_file=model_info.get('model_file')
    model_name=model_info.get('model_name')
    iter_freq=model_info.get('iter_freq')


    out_dir= dir_info.get('out_dir')
    train_dir= dir_info.get('train_dir')
    valid_dir= dir_info.get('valid_dir')
    test_dir= dir_info.get('test_dir')
    model_dir= dir_info.get('model_dir')
    cluster= dir_info.get('cluster')
    num_data_workers= dir_info.get('num_data_workers')

    wandb_update= wandb_parameters.get('wandb_update')
    wandb_project= wandb_parameters.get('wandb_project')
    wandb_ = wandb_parameters.get('wandb_update')
    wandb_key = wandb_parameters.get('wandb_key')


except FileNotFoundError:
    print(f"Error: Configuration file '{config_file}' not found.")
except yaml.YAMLError as e:
    print(f"Error parsing YAML file: {e}")

torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda)
use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{cuda}" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True



channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy","HBHE_energy", "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2" ,"Tob_1", "Tob_2"]


channels_used = [channel_list[ch] for ch in indices]
layers_names = ' | '.join(channels_used)


decay = f'{len(indices)}_ch_classifier_{model_name}_test'

if timestr == 'None':
    timestr=time.strftime("%Y-%m-%d-%H:%M:%S")
else:
    timestr=timestr


if not os.path.isdir(out_dir+'/'+decay):
    os.makedirs(out_dir+'/'+decay)


def mae_loss_wgtd(pred, true, wgt=1.):
    loss = wgt*(pred-true).abs().to(device)
    return loss.mean()

# huber loss
def huber(pred, true, delta):
    if (true-pred).abs().to(device) < delta:
        loss = 0.5*((true-pred)**2).to(device)
    else:
        loss = delta*(pred-true).abs().to(device) - 0.5*(delta**2)
    return loss.mean()


# log cosh loss
def logcosh(pred, true):
    #loss = torch.mean( torch.log( torch.cosh(y - y_hat) ))
    loss = torch.log(torch.cosh(pred - true)).to(device)
    return loss.mean()



print(">>>> Channels used for this model >>>  %s" %layers_names)
print(">>>> Device Used >>>  %s" %device)




networks = importlib.import_module(model_file)
if model_name=='ResNet':
    resnet = networks.ResNet(len(indices), resblocks, reslayers)
if model_name=='ResNet_BN':
    resnet = networks.ResNet_BN(len(indices), resblocks, reslayers)
if model_name=='ResNet_mapA':
    resnet = networks.ResNet_mapA(len(indices), resblocks, reslayers, 1)
if model_name=='ResNet_MultiChannel_conv':
    resnet = networks.ResNet_MultiChannel_conv(len(indices), resblocks, reslayers)
if model_name=='ResNet_MultiChannel_conv_with_map':
    resnet = networks.ResNet_MultiChannel_conv_with_map(len(indices), resblocks, reslayers)
resnet=resnet.to(device)

w_intr = iter_freq
def do_eval(resnet, val_loader, epoch, out_file):
    torch.cuda.empty_cache()
    loss_, acc_ = 0., 0.
    y_pred_, y_true_ = [], []
    now = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)
            iphi, ieta = data[2].to(device), data[3].to(device)

            iphi = iphi/360.
            ieta = ieta/140.
            logits = resnet([X, iphi, ieta])
            loss=  F.binary_cross_entropy_with_logits(logits, y)
            loss_ += loss.item()
            pred = logits.ge(0.).byte()
            acc_ += pred.eq(y.byte()).float().mean().item()
            y_pred_.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_true_.append(y.detach().cpu().numpy())


            if i % w_intr  == 0:

                print('Validation (%d/%d): Val loss:%f, acc:%f'%(i+1, len(test_loader), loss_/(i+1), acc_/(i+1) ))

            del logits
            # gc.collect()
        now = time.time() - now
        y_pred_ = np.concatenate(y_pred_)
        y_true_ = np.concatenate(y_true_)

        print('%d: Val loss:%f, acc%f'%(epoch, loss_/len(test_loader), np.mean(acc_)))
        output_dict = {}
        output_dict["y_true"] = y_true_
        output_dict["y_pred"] = y_pred_

        with open(f'{out_dir}/{decay}/{out_file}_test_data.pkl', "wb") as outfile:
          pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility
    del y_pred_
    del y_true_
    # gc.collect()

# MAIN #


load_model = glob.glob(f'{model_dir}/model_epoch{load_epoch}*.pkl')[0]#loading  model mannually
print('Loading weights from %s'%load_model)
checkpoint = torch.load(load_model)
resnet.load_state_dict(checkpoint['model_state_dict'])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
resnet.eval()
test_dirs = glob.glob(f'{test_dir}/*.h5')
# print(test_dirs)
for file_path in test_dirs:
    classification_dset = ClassifierDataset(file_path, selected_channels=indices, preload_size=32)
    n_test = len(classification_dset)
    test_indices = list(range(n_test))
    print('>> Samples in test : %s '%file_path)
    print('>> Number of samples in dataset: Total: %d '%n_test)
    test_sampler = ChunkedSampler(test_indices, chunk_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(classification_dset, batch_size=TEST_BATCH_SIZE, sampler=test_sampler, pin_memory=True, num_workers=num_data_workers)
    out_file_name = os.path.splitext(os.path.basename(file_path))[0]
    do_eval(resnet, test_loader, load_epoch, out_file_name)
