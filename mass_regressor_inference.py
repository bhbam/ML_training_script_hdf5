import sys, yaml, h5py, random, os, glob, gc, time, pickle
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from dataset_loder import *
import pandas as pd
from torch_resnet_concat import *
# import importlib
# import mplhep as hep
# import matplotlib.pyplot as plt
# import torch_resnet_concat as networks
def correction(mass_raw,a=-0.04922334  ,b=-1.25575772):
            # mass = m + polynomial_mode(m)
            mass = mass_raw + a*mass_raw + b
            return mass

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_random_seeds(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Correct way to specify GPU index
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("device used:-  ",  device)
torch.backends.cudnn.benchmark = True

# Choose the loss function dynamically
loss_type = "mae"  # Options: "mse", "mae", "huber", "logcosh"
loss_functions = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    "huber": nn.SmoothL1Loss(beta=0.5)

}

criterion = loss_functions[loss_type]
load_epoch = 100
reslayers= [8,16,32,64]
resblocks= 3
m0_scale = 14
num_data_workers= 1

# channel_list = ["Tracks_pt", "Tracks_dZSig", "Tracks_d0Sig", "ECAL_energy","HBHE_energy", "Pix_1", "Pix_2", "Pix_3", "Pix_4", "Tib_1", "Tib_2" ,"Tob_1", "Tob_2"]

n_test = -1
BATCH_SIZE=128


Mass = '14'
m_true = {'3p7':3.7, '14':14}.get(Mass, None)

model_dir ="/uscms/home/bbbam/nobackup/analysis_run3/Data_for_plots/mass_regression_test_data/jupyter_notebook_new/ResNet_mapA_mN1p2To22_unbaised"
models = ['13_ResNet_mapA_Tracks_pt_Tracks_dZSig_Tracks_d0Sig_ECAL_energy_HBHE_energy_Pix_1_Pix_2_Pix_3_Pix_4_Tib_1_Tib_2_Tob_1_Tob_2'
              ,'12_ResNet_mapA_Tracks_dZSig_Tracks_d0Sig_ECAL_energy_HBHE_energy_Pix_1_Pix_2_Pix_3_Pix_4_Tib_1_Tib_2_Tob_1_Tob_2'
              ,'11_ResNet_mapA_Tracks_pt_ECAL_energy_HBHE_energy_Pix_1_Pix_2_Pix_3_Pix_4_Tib_1_Tib_2_Tob_1_Tob_2'
              ,'10_ResNet_mapA_ECAL_energy_HBHE_energy_Pix_1_Pix_2_Pix_3_Pix_4_Tib_1_Tib_2_Tob_1_Tob_2'
              ,'5_ResNet_mapA_Tracks_pt_Tracks_dZSig_Tracks_d0Sig_ECAL_energy_HBHE_energy'
              ,'4_ResNet_mapA_Tracks_pt_Tracks_dZSig_Tracks_d0Sig_ECAL_energy'
               ]
out_dir = f"/uscms/home/bbbam/nobackup/analysis_run3/Data_for_plots/ResNet_mapA_jet_mult_test"
test_dir = glob.glob(f"/eos/uscms/store/user/bbbam/signals_jet_multi/IMG_HToAATo4Tau_Hadronic_signal_mass_14_GeV_with_jet_multi_test_combined_h5/IMG_HToAATo4Tau_Hadronic_signal_mass_14_GeV_with_jet_mult.h5")[0]
# test_dir = glob.glob(f"/eos/uscms/store/user/bbbam/signal_combined_from_ruchi_May_1_2025_h5_original_h5/IMG_HToAATo4Tau_Hadronic_signal_mass_14_GeV_combined.h5")[0]
test_indices = list(range(len(test_dir)))

channels = [[0,1,2,3,4,5,6,7,8,9,10,11,12]
            ,[1,2,3,4,5,6,7,8,9,10,11,12]
            ,[0,3,4,5,6,7,8,9,10,11,12]
            ,[3,4,5,6,7,8,9,10,11,12]
            ,[0,1,2,3,4]
            ,[0,1,2,3]
           ]

# for ch, model in enumerate(models):
ch, model = 0, models[0]
# print("channels:  ", channels[ch], "models:  ", model)
test_dset =  RegressionDataset_with_channel_selector_valid(test_dir, selected_channels=channels[ch], preload_size=32)
n_total_test = len(test_dset)

if n_test !=-1:
    test_indices = list(range(n_test))
else:
    test_indices = list(range(n_total_test))
n_test = len(test_indices)

print(f"Total test sample : {n_total_test} only used: {n_test} ---> {n_test/n_total_test*100} %")
w_iter_freq = n_test//10

test_sampler = ChunkedSampler(test_indices, chunk_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dset, batch_size=BATCH_SIZE, sampler=test_sampler, pin_memory=True, num_workers=num_data_workers)


resnet = ResNet_mapA(in_channels=len(channels[ch]), nblocks=resblocks, fmaps=reslayers, alpha=1)
resnet=resnet.to(device)
load_model = glob.glob(f'{model_dir}/{model}/MODELS/model_epoch{load_epoch}*')[0]#loading  model mannually
print('Loading weights from %s'%load_model)
checkpoint = torch.load(load_model, map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
with torch.no_grad():
        m_pred_, m_true_, mae_, mre_, jet_mass_, jet_pt_, jet_mult_ = [], [], [], [], [],[],[]
        loss_ =0
        for i, data in enumerate(test_loader):
            X, am = data[0].to(device), data[1].to(device)
            iphi, ieta = data[2].to(device), data[3].to(device)
            am = transform_y(am, m0_scale)
            # print("am------------------------", am)
            iphi = iphi/360.
            ieta = ieta/140.
            logits = resnet([X, iphi, ieta])
            # print("logits------------------------", logits)
            loss = criterion(logits, am).item()
            loss_ += loss
            logits, am = inv_transform_y(logits, m0_scale), inv_transform_y(am, m0_scale)
            mae = (logits-am).abs()
            mre = (((logits-am).abs())/am)
            m_pred_.append(logits.detach().cpu().numpy())
            m_true_.append(am.detach().cpu().numpy())
            mae_.append(mae.detach().cpu().numpy())
            mre_.append(mre.detach().cpu().numpy())
            jet_mass_.append(data[4])
            jet_pt_.append(data[5])
            jet_mult_.append(data[6])

            if i % w_iter_freq  == 0:


                print('Validation (%d/%d): Val loss:%f, mae:%f, mre:%f'%(i+1, len(test_loader), loss, mae.mean().item(), mre.mean().item() ))

            del logits

        # now = time.time() - now
        m_true_ = np.concatenate(m_true_)
        m_pred_ = np.concatenate(m_pred_)
        mae_ = np.concatenate(mae_)
        mre_ = np.concatenate(mre_)
        jet_mass_ = np.concatenate( jet_mass_)
        jet_pt_ = np.concatenate(jet_pt_)
        jet_mult_ = np.concatenate(jet_mult_)



        output_dict = {}
        output_dict["m_true"] = m_true_
        output_dict["m_pred"] = correction(m_pred_)
        output_dict["jet_mass"] = jet_mass_
        output_dict["jet_pt"] = jet_pt_
        output_dict["jet_mult"] = jet_mult_




        print('%d: Val loss:%f, mae:%f, mre:%f'%(load_epoch, loss_/len(test_loader), np.mean(mae_), np.mean(mre_)))

        os.makedirs(f'{out_dir}', exist_ok=True)
        with open(f'{out_dir}/signal_{Mass}_epoch_{load_epoch}_inference_data_from_h5.pkl', "wb") as outfile:
              pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility
        print(f'---Done----{out_dir}/signal_{Mass}_epoch_{load_epoch}_inference_data_.pkl saved.')
