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
    signal_mass= scales.get('signal_mass')


    run_logger=model_info.get('run_logger')
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



np.random.seed(random_seed)
os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda)
use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{cuda}" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True



m0_scale    = torch.tensor(m0_scale)
channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy","HBHE_energy", "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2" ,"Tob_1", "Tob_2"]


channels_used = [channel_list[ch] for ch in indices]
layers_names = ' | '.join(channels_used)


decay = f'{len(indices)}_ch_massregressor_{model_name}'

if timestr == 'None':
    timestr=time.strftime("%Y-%m-%d-%H:%M:%S")
else:
    timestr=timestr


file_test = glob.glob(f'{test_dir}')[0]


test_dset = RegressionDataset(file_test, preload_size=BATCH_SIZE)
n_total_test = len(test_dset)


if n_test !=-1:
    test_indices = list(range(n_test))
    random.shuffle(test_indices)
else:
    test_indices = list(range(n_total_test))
    random.shuffle(test_indices)

n_test = len(test_indices)

if run_logger:
    for d in ['INFERENCE_DATA_test']:
        if not os.path.isdir(out_dir+'/'+decay+'/%s'%d):
            os.makedirs(out_dir+'/'+decay+'/%s'%d)
    f = open(out_dir +'/'+ decay+'/%s_timestamp_%s.log'%(decay, timestr), 'w')
else:
    f=''

def logger(s):
    global f, run_logger
    print(s)
    if run_logger:
        f.write('%s\n'%str(s))

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

def set_scheduler(optimizer, scheduler_, mode, patience, lr_init, lr_factor):
    """ set the lr scheduler """
    if  scheduler_ == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, patience=patience, verbose=True, min_lr=lr_init, factor=lr_factor)
    elif scheduler_ == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=patience)
    else:
        scheduler = None
    return scheduler

def set_optimizer(optimizer_,lr_init):
    """ set the optimizer """
    if optimizer_ == 'Adam':
        optimizer = optim.Adam(resnet.parameters(), lr=lr_init)
    elif optimizer_ == 'SGD':
        optimizer = optim.SGD(resnet.parameters(), lr=lr_init, momentum=0.9)
    return optimizer

logger(">>>> Channels used for this model >>>  %s" %layers_names)
logger(">>>> Device Used >>>  %s" %device)
logger('>> Number of samples in dataset: Total:  test %d'% n_total_test)
logger('>> Number of samples used for test:  Val: %d'% n_test)

test_sampler = ChunkedSampler(test_indices, chunk_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dset, batch_size=BATCH_SIZE, sampler=test_sampler, pin_memory=True, num_workers=num_data_workers)


networks = importlib.import_module(model_file)
if model_name=='ResNet':
    resnet = networks.ResNet(len(indices), resblocks, reslayers)
resnet=resnet.to(device)


w_intr = iter_freq

def do_eval(resnet, test_loader, epoch):
    torch.cuda.empty_cache()
    loss_ = 0.
    m_pred_, m_true_, mae_, mre_, m0_ = [], [], [], [], []
    now = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X, am = data[0].to(device), data[1].to(device)
            iphi, ieta = data[2].to(device), data[3].to(device)
            am = transform_y(am, m0_scale)
            iphi = iphi/360.
            ieta = ieta/140.
            logits = resnet([X, iphi, ieta])
            loss= mae_loss_wgtd(logits, am).item()
            loss_ += loss
            logits, am = inv_transform(logits,m0_scale), inv_transform(am,m0_scale)
            mae = (logits-am).abs()
            mre = (((logits-am).abs())/am)
            m_pred_.append(logits.detach().cpu().numpy())
            m_true_.append(am.detach().cpu().numpy())
            mae_.append(mae.detach().cpu().numpy())
            mre_.append(mre.detach().cpu().numpy())


            if i % w_intr  == 0:

                logger('Testing (%d/%d): Train loss:%f, mae:%f, mre:%f'%(i+1, len(test_loader), loss, mae.mean().item(), mre.mean().item() ))

            del logits
            # gc.collect()
        now = time.time() - now
        m_true_ = np.concatenate(m_true_)
        m_pred_ = np.concatenate(m_pred_)
        mae_    = np.concatenate(mae_)
        mre_    = np.concatenate(mre_)





        logger('%d: Val loss:%f, mae:%f, mre:%f'%(epoch, loss_/len(test_loader), np.mean(mae_), np.mean(mre_)))
        score_str = 'epoch%d__mae%.4f'%(epoch, np.mean(mae_))

        output_dict = {}
        output_dict["m_true"] = m_true_
        output_dict["m_pred"] = m_pred_
        output_dict["mae"] = mae_
        output_dict["mre"] = mre_


        with open(f'{out_dir}/{decay}/INFERENCE_DATA_test/Mass_{signal_mass}_{score_str}_inference_data.pkl', "wb") as outfile:
          pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility
        mae_retun = np.mean(mae_)
        del m_pred_
        del m_true_
        del mae_
        del mre_
        # gc.collect()
        return mae_retun

load_model = model_dir #loading  model mannually
logger('Loading weights from %s'%load_model)
if torch.cuda.is_available():
    checkpoint = torch.load(load_model)
else:
    checkpoint = torch.load(load_model, map_location=torch.device('cpu'))
resnet.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

logger(f">>>>>>> Run Testing for signal mass {signal_mass} <<<<<<<<")

resnet.eval()
do_eval(resnet, test_loader, load_epoch)
logger(">>>>>>> End Testing <<<<<<<<")
