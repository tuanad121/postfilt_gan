from __future__ import print_function
import argparse
import os
import random
#import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from data_loader import get_loader, prepare_normalizer
from utils import plot_feats, read_binary_file
from models import define_netCNN


def train(netG, data_loader, opt):
    # cost criterion
    # criterion = nn.BCELoss() # normal gan 
    criterion = nn.MSELoss() # lsgan

    if opt.cuda:
        netG.cuda()
        criterion.cuda()
    
    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    print('batch size =', opt.batchSize)
    for epoch in range(opt.niter):
        # store mini-batch data in list
        for i, (real_data, pred_data) in enumerate(data_loader):
            netG.zero_grad()
            if opt.cuda:
                pred_data = pred_data.cuda()
                real_data = real_data.cuda()

            pred_data = Variable(pred_data)
            real_data = Variable(real_data)

            # train with fake 
            fake = netG(pred_data)
            # add the residual to the tts predicted data 
            fake = fake + pred_data
            
            # errG = criterion(output, label)
            errG = criterion(real_data, fake)
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_G: %.4f'
                %(epoch, opt.niter, i, len(data_loader),
                errG.item()))
            
            del errG, real_data, pred_data, fake
            torch.cuda.empty_cache()

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netCNN_epoch_%d.pth' %(opt.outf, epoch))

def test(netG, opt):
    assert opt.netG != ''
    test_dir = opt.testdata_dir
    for f in os.listdir(test_dir):
        fname, ext = os.path.splitext(f)
        if ext == '.cmp':
            print(fname)
            cmp_file = os.path.join(test_dir, f)
            ac_data = read_binary_file(cmp_file, dim=47)
            ac_data = torch.FloatTensor(ac_data)
            noise = torch.FloatTensor(ac_data.size(0), nz)
            if opt.cuda:
                ac_data, noise = ac_data.cuda(), noise.cuda()
            ac_data = Variable(ac_data)
            noise = Variable(noise)
            noise.data.normal_(0, 1)
            generated_pulses = netG(noise, ac_data)
            generated_pulses = generated_pulses.data.cpu().numpy()
            generated_pulses = generated_pulses.reshape(ac_data.size(0), -1)
            out_file = os.path.join(test_dir, fname + '.pls')
            with open(out_file, 'wb') as fid:
                generated_pulses.tofile(fid)    


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--voiceName', required=True, help='nick | jenny ')
    parser.add_argument('--mode', required=True, type=str, help='train | test')
    parser.add_argument('--xFilesList', required=True, help='path to input files list')
    parser.add_argument('--yFilesList', required=True, help='path to output files list')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--mgcDim', type=int, default=40, help='mel-cepstrum dimension')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--testdata_dir', type=str, help='path to test data')
    opt = parser.parse_args()
    print(opt)
    print(torch.__version__)
    device = torch.device("cuda:1" if opt.cuda else "cpu")
  # prepare the data loader
    x_files_list_file = opt.xFilesList
    y_files_list_file = opt.yFilesList
    in_dim = opt.mgcDim
    out_dim = opt.mgcDim

    with open(x_files_list_file, 'r') as fid:
        x_files_list = [l.strip() for l in fid.readlines()]

    with open(y_files_list_file, 'r') as fid:
        y_files_list = [l.strip() for l in fid.readlines()]
    
    x_normalizer = prepare_normalizer(x_files_list, in_dim)
    y_normalizer = prepare_normalizer(y_files_list, out_dim)

    data_loader = get_loader(x_files_list, y_files_list, 
                            in_dim, out_dim, opt.batchSize, False, 10, x_normalizer, y_normalizer)  

    # prepare the output directories
    try:
        os.makedirs(opt.outf)
        os.makedirs(os.path.join(opt.outf, 'figures'))
    except OSError:
        pass

    # if manual seed is not provide then pick one randomly
    if opt.manualSeed is None:
        opt.manualSeed  = random.randint(1, 10000)
    print('Random Seed: ', opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.enabled = False
    cudnn.benchmark = False

    # define the generator 
    netG = define_netCNN(device=device)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    if opt.mode == 'train':
        train(netG, data_loader, opt)
    elif opt.mode == 'test':
        test(netG, opt)
    else:
        print('Mode must be either train or test only')
