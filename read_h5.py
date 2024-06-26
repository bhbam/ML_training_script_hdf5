import glob
import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

file = '/scratch/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2_normalized_NAN_removed_train.h5'
data = h5py.File(file, 'r')
num_images = len(data['am'])
print("Total events  ", num_images)

start_idx = 0
batch_size = 20000
# num_images = 100
# Example batch processing

m_higher = 0
mass_array = []

for start_idx in tqdm(range(0, num_images, batch_size)):
    end_idx = min(start_idx + batch_size, num_images)

    am_batch = data["am"][start_idx:end_idx, :]
    mass_array.append(am_batch)
    am_mask = am_batch > 17.2
    am_batch = np.where(am_mask, am_batch, np.nan)
    size_channel = np.count_nonzero(~np.isnan(am_batch))
    m_higher = m_higher + size_channel

mass_array = np.concatenate(mass_array, axis=0).flatten()
mass_array = mass_array[~np.isnan(mass_array)]

plt.hist(mass_array, edgecolor='black')
plt.title('Histogram of Mass Values')
plt.xlabel('Mass')
plt.ylabel('Frequency')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('mass_histogram.png')

plt.show()

print("Mass higher than 17.2 GeV:", m_higher, "--->", m_higher / num_images * 100, '%')
