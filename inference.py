from pathlib import Path

from models import define_netG
from utils import read_binary_file

import torch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

device = torch.device("cpu")
netG = define_netG(in_ch=2, device=device)
netG.load_state_dict(torch.load('netG_epoch_24.pth'))

nat_dir_path = Path('/Users/dintu/zalo_ai/ZaloAi/mgc')
syn_dir_path = Path('/Users/dintu/zalo_ai/ZaloAi/syn_feat')
for file in syn_dir_path.glob('*.mgc'):
    print(file)
    syn_mgc, _ = read_binary_file(file, dim=25)
    print(syn_mgc.shape)
    syn_mgc = syn_mgc.T
    syn_mgc= syn_mgc.reshape(1,1,syn_mgc.shape[1], syn_mgc.shape[0])

    nat_mgc, _ = read_binary_file(nat_dir_path.joinpath(file.name), dim=25)
    print(nat_mgc.shape)

    syn_mgc = torch.FloatTensor(syn_mgc)
    noise = torch.FloatTensor(syn_mgc.size()).normal_(0,1)

    residual = netG(noise, syn_mgc)
    enh_mgc = residual + syn_mgc
    enh_mgc = enh_mgc.detach().numpy()[0,0,:,:]
    if 1:
        plt.subplot(3,1,1)
        plt.imshow(syn_mgc[0,0,:,:], origin='bottom', aspect='auto', cmap=plt.cm.gray)
        plt.subplot(3,1,2)
        plt.imshow(enh_mgc, origin='bottom', aspect='auto', cmap=plt.cm.gray)
        plt.subplot(3,1,3)
        plt.imshow(nat_mgc.T, origin='bottom', aspect='auto', cmap=plt.cm.gray)
        plt.show()
    

