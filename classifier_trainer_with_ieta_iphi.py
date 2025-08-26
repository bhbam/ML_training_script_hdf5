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


decay = f'{len(indices)}_ch_classifier_{model_name}'

if timestr == 'None':
    timestr=time.strftime("%Y-%m-%d-%H:%M:%S")
else:
    timestr=timestr

file_path = glob.glob(f'{train_dir}')[0]

classification_dset = ClassifierDataset(file_path, selected_channels=indices, preload_size=32)
n_total = len(classification_dset)
if load_epoch == 0:
    if n_train != -1:
        total_indices = list(range(n_train))
        random.shuffle(total_indices)
        train_size = int(0.9 * n_train)
        train_indices, valid_indices = total_indices[:train_size], total_indices[train_size:]

    else:
        total_indices = list(range(n_total))
        random.shuffle(total_indices)
        train_size = int(0.9 * n_total)
        train_indices, valid_indices = total_indices[:train_size], total_indices[train_size:]

        np.savez(f"{decay}_train_valid_indices.npz", train_indices=train_indices, valid_indices=valid_indices)

else:
    data_indices = np.load(f'{decay}_train_valid_indices.npz')
    train_indices = data_indices['train_indices']
    valid_indices = data_indices['valid_indices']

n_train = len(train_indices)
n_val = len(valid_indices)

if run_logger:
    for d in ['MODELS','INFERENCE_DATA']:
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
logger('>> Number of samples in dataset: Total: %d '%n_total)
logger('>> Number of samples used for training: Train: %d + Val: %d'%(n_train, n_val))

train_sampler = ChunkedSampler(train_indices, chunk_size=BATCH_SIZE, shuffle=True)
train_loader = DataLoader(classification_dset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, num_workers=num_data_workers)

val_sampler = ChunkedSampler(valid_indices, chunk_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(classification_dset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, num_workers=num_data_workers)


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

optimizer = set_optimizer(optimizer_,lr_init)
scheduler = set_scheduler(optimizer, scheduler_, scheduler_mode, patience, lr_init, lr_factor)


if wandb_:
    wandb.login(key=wandb_key)
    wandb.init(
        project =wandb_project,
        name = f"{decay}_{timestr}{cluster}"
    )

    wandb.watch(resnet, log_freq=500)


w_intr = iter_freq

def do_eval(resnet, val_loader, roc_auc_best, epoch):
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
                if wandb_:wandb.log({"val_loss": loss_/(i+1)})
                if wandb_:wandb.log({"val_acc": acc_/(i+1)})

                logger('Validation (%d/%d): Val loss:%f, acc:%f'%(i+1, len(val_loader), loss_/(i+1), acc_/(i+1) ))

            del logits
            # gc.collect()
        now = time.time() - now
        y_pred_ = np.concatenate(y_pred_)
        y_true_ = np.concatenate(y_true_)





        logger('%d: Val loss:%f, acc%f'%(epoch, loss_/len(val_loader), np.mean(acc_)))
        score_str = 'epoch%d_auc%.4f'%(epoch, roc_auc_best)

        fpr, tpr, _ = roc_curve(y_true_, y_pred_)
        roc_auc = auc(fpr, tpr)
        s = "VAL ROC AUC: %f"%(roc_auc)
        if wandb_:wandb.log({"val_auc": roc_auc})
        logger(s)

        # scheduler.step(loss_/len(val_loader))
        # print(optimizer.param_groups[0]['lr'])



        if roc_auc > roc_auc_best and run_logger:
            roc_auc_best = roc_auc
            logger('Best ROC AUC:%.4f\n'%roc_auc_best)
            score_str = 'epoch%d_auc%.4f'%(epoch, roc_auc_best)
            filename  = f'{out_dir}/{decay}/MODELS/model_{score_str}.pkl'
            loss_value = loss_/len(val_loader)
            model_dict = {'model_state_dict': resnet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'epoch' : epoch, 'loss': loss_value}
            torch.save(model_dict, filename)

            output_dict = {}
            output_dict["y_true"] = y_true_
            output_dict["y_pred"] = y_pred_
            output_dict["fpr"] = fpr
            output_dict["tpr"] = tpr


            with open(f'{out_dir}/{decay}/INFERENCE_DATA/{score_str}_inference_data.pkl', "wb") as outfile:
              pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility
        del y_pred_
        del y_true_
        # gc.collect()
        scheduler.step()
        return roc_auc_best

# MAIN #

if load_epoch != 0:
    load_model = glob.glob(f'{out_dir}/{decay}/MODELS/model_epoch{load_epoch}*')[0]#loading  model mannually
    logger('Loading weights from %s'%load_model)
    checkpoint = torch.load(load_model)
    resnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if new_lr != 0:
            logger(' - OLD LR = %f'%optimizer.param_groups[0]['lr'])
            optimizer.param_groups[0]['lr'] = new_lr
            logger(' - NEW LR = %f'%optimizer.param_groups[0]['lr'])


logger(">> Training <<<<<<<<")
roc_auc_best =0.5
for e in range(epochs):
    loss_t, acc_ = 0., 0.
    epoch = e+1+load_epoch
    epoch_wgt = 0.
    n_trained = 0
    logger('>> Epoch %d <<<<<<<<'%(epoch))

    # Run training
    # scheduler.step(roc_auc_best)
    resnet.train()
    now = time.time()
    for i, data in enumerate(train_loader):
        X, y= data[0].to(device), data[1].to(device)
        iphi, ieta = data[2].to(device), data[3].to(device)

        with torch.no_grad():
            iphi = iphi/360.
            ieta = ieta/140.

        

        optimizer.zero_grad()
        logits = resnet([X, iphi, ieta])
        loss=  F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        epoch_wgt += len(y)
        loss_t += loss.item()
        n_trained += 1
        pred = logits.ge(0.).byte()
        acc_ += pred.eq(y.byte()).float().mean().item()
        if i % w_intr  == 0:
            logger('%d: (%d/%d) y_pred: %s...'%(epoch, i+1, len(train_loader), str(np.squeeze(torch.sigmoid(logits).tolist()[:5]))))
            logger('%d: (%d/%d) y_true: %s...'%(epoch, i+1, len(train_loader), str(np.squeeze(y.tolist()[:5]))))
            logger('%d: (%d/%d) Train loss:%f, acc:%f'%(epoch, i+1, len(train_loader), loss.item(), acc_))
            if wandb_:wandb.log({"train_loss": loss_t/(i+1)})
            if wandb_:wandb.log({"train_acc": acc_/(i+1)})

    now = time.time() - now
    logger('%d: Train time:%.2fs in %d steps for N:%d, wgt: %.f'%(epoch, now, len(train_loader), n_trained, epoch_wgt))
    logger('%d: Train loss:%f, acc:%f'%(epoch, loss_t/(i+1), acc_/(i+1)))


    # Run Validation
    resnet.eval()
    val_mae = do_eval(resnet, val_loader, roc_auc_best, epoch)
    curr_lr = scheduler._last_lr[0]
    if wandb_:
        wandb.log({"Epoch": epoch,
                    "resblocks" : resblocks,
                    "m0_scale": m0_scale,
                    "Lr": curr_lr,
                    "n_total": n_total,
                    "n_train": n_train,
                    "n_test": n_val,
                    "channels": len(indices),
                    "lr_init": lr_init,
                    "lr_factor": lr_factor,
                    "patience": patience,
                    "epochs": epochs,
                    "initial_epoch": load_epoch,
                    "BATCH_SIZE": BATCH_SIZE,
                    }
                    )
    # gc.collect()

if wandb_:wandb.finish()


if run_logger:f.close()
