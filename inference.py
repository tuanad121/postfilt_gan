from pathlib import Path
import random

from models import define_netG
from utils import read_binary_file

import torch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pysptk.sptk import mgc2sp

device = torch.device("cpu")
netG = define_netG(in_ch=2, device=device)
netG.load_state_dict(torch.load('netG_epoch_22.pth'))

nat_dir_path = Path('/Users/dintu/zalo_ai/postfilt_gan/mgc')
syn_dir_path = Path('/Users/dintu/zalo_ai/postfilt_gan/mgc_gen')


def prepare_normalizer(list_paths, dim):
        dataset = []
        for file_path in list_paths:
                try:
                        data, _ = read_binary_file(file_path, dim)
                        dataset.append(data)
                except FileNotFoundError:
                        print(FileNotFoundError)
        dataset = np.concatenate(dataset)
        scaler = StandardScaler().fit(dataset)
        del dataset
        return scaler

with open('gen_files.list', 'r') as fid:
        gen_files_list = [l.strip() for l in fid.readlines()]

with open('ref_files.list', 'r') as fid:
        ref_files_list = [l.strip() for l in fid.readlines()]

gen_normalizer = prepare_normalizer(gen_files_list, dim=40)
ref_normalizer = prepare_normalizer(ref_files_list, dim=40)
print(gen_normalizer.mean_)
assert len(list(syn_dir_path.glob('*.mgc'))) > 0
for file in syn_dir_path.glob('*.mgc'):
        print(file)
        
        syn_mgc_ori, _ = read_binary_file(file, dim=40)
        rnd = random.randint(0, syn_mgc_ori.shape[0]-40)
        print(rnd)

        syn_mgc = gen_normalizer.transform(syn_mgc_ori)
        syn_mgc = syn_mgc.T
        print(syn_mgc.shape)
        syn_mgc= syn_mgc.reshape(1,1,syn_mgc.shape[0], syn_mgc.shape[1])

        nat_mgc, _ = read_binary_file(nat_dir_path.joinpath(file.name), dim=40)
        # nat_mgc = ref_normalizer.transform(nat_mgc)
        # nat_sp = mgc2sp(nat_mgc.astype('float64'), alpha=0.42, gamma=0.0, fftlen=1024).T
        if 0:
                np.savetxt(file.name+'a', nat_mgc.flatten())
        nat_mgc = nat_mgc.T
        print(nat_mgc.shape)

        syn_mgc = torch.FloatTensor(syn_mgc)
        noise = torch.FloatTensor(syn_mgc.size()).normal_(0,1)

        residual = netG(noise, syn_mgc)
        print(residual.shape)
        enh_mgc = residual + syn_mgc
        enh_mgc = enh_mgc.detach().numpy()[0,0,:,:]
        enh_mgc = ref_normalizer.inverse_transform(enh_mgc.T).T

        # syn_mgc = syn_mgc.detach().numpy()[0,0,:,:]

        syn_mgc_ori = syn_mgc_ori.T
        if 0:
                plt.subplot(3,1,1)
                plt.imshow(syn_mgc_ori[:, rnd: rnd + 40], origin='bottom', aspect='auto') #, cmap=plt.cm.gray)
                plt.subplot(3,1,2)
                plt.imshow(enh_mgc[:, rnd: rnd + 40], origin='bottom', aspect='auto') #, cmap=plt.cm.gray)
                plt.subplot(3,1,3)
                plt.imshow(nat_mgc[:, rnd: rnd + 40], origin='bottom', aspect='auto') #, cmap=plt.cm.gray)
                plt.show()
        
        if 1:
                np.savetxt(file.name+'a', enh_mgc.T.flatten())
        
        # if 0:
        #         enh_sp = mgc2sp(enh_mgc.T.astype('float64'), alpha=0.42, gamma=0, fftlen=1024).T
                
        #         plt.subplot(3,1,1)
        #         plt.imshow(20*np.log10(np.abs(syn_sp)), origin='bottom', aspect='auto', cmap=plt.cm.gray)
        #         plt.subplot(3,1,2)
        #         plt.imshow(20*np.log10(np.abs(enh_sp)), origin='bottom', aspect='auto', cmap=plt.cm.gray)
                
        #         plt.subplot(3,1,3)
        #         plt.imshow(20*np.log10(np.abs(nat_sp)), origin='bottom', aspect='auto', cmap=plt.cm.gray)
        #         plt.show()

    

