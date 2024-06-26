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

file_path = glob.glob(f'{train_dir}')[0]

reg_dset = RegressionDataset(file_path, preload_size=BATCH_SIZE)
n_total = len(reg_dset)
if load_epoch == 0:
    if n_train != -1:
        total_indices = list(range(n_train))
        train_size = int(0.9 * n_train)
        train_indices, valid_indices = total_indices[:train_size], total_indices[train_size:]

    else:
        total_indices = list(range(n_total))
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
train_loader = DataLoader(reg_dset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, num_workers=num_data_workers)

val_sampler = ChunkedSampler(valid_indices, chunk_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(reg_dset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, num_workers=num_data_workers)


networks = importlib.import_module(model_file)
if model_name=='ResNet':
    resnet = networks.ResNet(len(indices), resblocks, reslayers)
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
mae_best = 10000.
def do_eval(resnet, val_loader, mae_best, epoch):
    torch.cuda.empty_cache()
    loss_ = 0.
    m_pred_, m_true_, mae_, mre_, m0_ = [], [], [], [], []
    now = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
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
                if wandb_:wandb.log({"val_loss": loss_/(i+1)})
                if wandb_:wandb.log({"val_mae":np.mean(np.concatenate(mae_))})

                logger('Validation (%d/%d): Train loss:%f, mae:%f, mre:%f'%(i+1, len(val_loader), loss, mae.mean().item(), mre.mean().item() ))

            del logits
            # gc.collect()
        now = time.time() - now
        m_true_ = np.concatenate(m_true_)
        m_pred_ = np.concatenate(m_pred_)
        mae_    = np.concatenate(mae_)
        mre_    = np.concatenate(mre_)





        logger('%d: Val loss:%f, mae:%f, mre:%f'%(epoch, loss_/len(val_loader), np.mean(mae_), np.mean(mre_)))
        score_str = 'epoch%d__mae%.4f'%(epoch, np.mean(mae_))

        scheduler.step(loss_/len(val_loader))
        print(optimizer.param_groups[0]['lr'])



        if run_logger and mae.mean().item() < mae_best:
            mae_best = mae.mean().item()
            filename  = f'{out_dir}/{decay}/MODELS/model_{score_str}.pkl'
            loss_value = loss_/len(val_loader)
            model_dict = {'model_state_dict': resnet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'epoch' : epoch, 'loss': loss_value}
            torch.save(model_dict, filename)

            output_dict = {}
            output_dict["m_true"] = m_true_
            output_dict["m_pred"] = m_pred_
            output_dict["mae"] = mae_
            output_dict["mre"] = mre_


            with open(f'{out_dir}/{decay}/INFERENCE_DATA/{score_str}_inference_data.pkl', "wb") as outfile:
              pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility
        mae_retun = np.mean(mae_)
        del m_pred_
        del m_true_
        del mae_
        del mre_
        # gc.collect()
        return mae_retun

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
for e in range(epochs):
    loss_t = 0.
    mae_t = 0.
    mre_t =0.
    epoch = e+1+load_epoch
    epoch_wgt = 0.
    n_trained = 0
    logger('>> Epoch %d <<<<<<<<'%(epoch))

    # Run training
    scheduler.step(mae_best)
    resnet.train()
    now = time.time()
    for i, data in enumerate(train_loader):
        X, am = data[0].to(device), data[1].to(device)
        iphi, ieta = data[2].to(device), data[3].to(device)

        with torch.no_grad():
            am = transform_y(am, m0_scale)
            iphi = iphi/360.
            ieta = ieta/140.

        optimizer.zero_grad()
        logits = resnet([X, iphi, ieta])
        loss = mae_loss_wgtd(logits, am)
        loss.backward()
        optimizer.step()
        epoch_wgt += len(am)
        loss_t += loss.item()
        n_trained += 1
        logits, am = inv_transform(logits,m0_scale), inv_transform(am,m0_scale)
        mae =  (logits-am).abs().mean()
        mre = (((logits-am).abs())/am).mean()
        mae_t += mae.item()
        mre_t += mre.item()
        if i % w_intr  == 0:
            logger('%d: (%d/%d) m_pred: %s...'%(epoch, i+1, len(train_loader), str(np.squeeze(logits.tolist()[:5]))))
            logger('%d: (%d/%d) m_true: %s...'%(epoch, i+1, len(train_loader), str(np.squeeze(am.tolist()[:5]))))
            logger('%d: (%d/%d) Train loss:%f, mae:%f, mre:%f'%(epoch, i+1, len(train_loader), loss.item(), mae.item(), mre.item() ))
            if wandb_:wandb.log({"train_loss": loss_t/(i+1)})
            if wandb_:wandb.log({"train_mae": mae_t/(i+1)})

    now = time.time() - now
    logger('%d: Train time:%.2fs in %d steps for N:%d, wgt: %.f'%(epoch, now, len(train_loader), n_trained, epoch_wgt))
    logger('%d: Train loss:%f, mae:%f, mre:%f'%(epoch, loss_t/(i+1), mae_t/(i+1), mre_t/(i+1) ))


    # Run Validation
    resnet.eval()
    val_mae = do_eval(resnet, val_loader, mae_best, epoch)
    # scheduler.step(val_mae)
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
