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
from models import define_netD, define_netG


def train(netD_A, netD_B, netG_AB, netG_BA, data_loader, opt, device):
    label = torch.FloatTensor(1)
    label = Variable(label, requires_grad=False)
    real_label = 1
    fake_label = 0

    # cost criterion
    # criterion = nn.BCELoss() # normal gan 
    criterion = nn.MSELoss() # lsgan

    if opt.cuda:
        netD_A.cuda(device)
        netD_B.cuda(device)
        netG_AB.cuda(device)
        netG_BA.cuda(device)
        criterion.cuda(device)
    
    # setup optimizer
    d_params = list(netD_A.parameters()) + list(netD_B.parameters())
    optimizerD = optim.Adam(d_params, lr=0.0001, betas=(0.5, 0.999))

    g_params = list(netG_AB.parameters()) + list(netG_BA.parameters())
    optimizerG = optim.Adam(g_params, lr=0.0002, betas=(0.5, 0.999))

    print('batch size =', opt.batchSize)
    for epoch in range(opt.niter):
        # store mini-batch data in list

        cycle_loss_lambda = 10
        identity_loss_lambda = 5

        for i, (real_data, pred_data) in enumerate(data_loader):
            # print(real_data.shape, pred_data.shape)
            #################################
            # (1) Updata G network
            #################################
            # A is pred_data and B is real_data
            # crop the tensor to fixed size
            rand_int = random.randint(0,real_data.size(-1) - opt.mgcDim)
            real_data_crop = real_data[:,:,:,rand_int:rand_int+opt.mgcDim]
            pred_data_crop = pred_data[:,:,:,rand_int:rand_int+opt.mgcDim]
            # label = torch.full((real_data.size(0),), real_label)
            # print(f'shape of real_data_crop {real_data_crop.shape}')
            noise = torch.FloatTensor(real_data.size()).normal_(0,1)
            if opt.cuda:
                real_data = real_data.cuda(device)
                pred_data = pred_data.cuda(device)
                real_data_crop = real_data_crop.cuda(device)
                pred_data_crop = pred_data_crop.cuda(device)
                noise = noise.cuda()

            pred_data = Variable(pred_data)
            real_data = Variable(real_data)
            real_data_crop = Variable(real_data_crop)
            pred_data_crop = Variable(pred_data_crop)
            
            fake_B = netG_AB(noise, pred_data)
            fake_B = fake_B + pred_data
            cycle_A = netG_BA(noise, fake_B)
            cycle_A = cycle_A + fake_B

            fake_A = netG_BA(noise, real_data)
            fake_A = fake_A + real_data
            cycle_B = netG_AB(noise, fake_A)
            cycle_B = cycle_B + fake_A

            identity_A = netG_BA(noise, pred_data)
            identity_A = identity_A + pred_data
            identity_B = netG_AB(noise, real_data)
            identity_B = identity_B + real_data
            
            fake_A_crop = fake_A[:,:,:,rand_int:rand_int+opt.mgcDim]
            fake_B_crop = fake_B[:,:,:,rand_int:rand_int+opt.mgcDim]

            d_fake_A = netD_A(fake_A_crop)
            d_fake_B = netD_B(fake_B_crop)

            # Generator Cycle Loss
            cycleLoss = torch.mean(
                torch.abs(pred_data - cycle_A)) + torch.mean(torch.abs(real_data - cycle_B))
            
            # Generator Identity Loss
            identiyLoss = torch.mean(
                    torch.abs(pred_data - identity_A)) + torch.mean(torch.abs(real_data - identity_B))

            # Generator Loss
            generator_loss_A2B = torch.mean((real_label - d_fake_B) ** 2)
            generator_loss_B2A = torch.mean((real_label - d_fake_A) ** 2)

            # errG = criterion(output, label)
            g_loss = generator_loss_A2B + generator_loss_B2A + \
                    cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss

            # Backprop for Generator
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            ############################
            # (2) Update D network
            ############################

            # train with real
            d_real_A = netD_A(pred_data_crop)
            d_real_B = netD_B(real_data_crop)

            generated_A = netG_BA(noise, real_data)
            generated_A = generated_A + real_data
            d_fake_A = netD_A(generated_A[:,:,:,rand_int:rand_int+opt.mgcDim])
            
            generated_B = netG_AB(noise, pred_data)
            generated_B = generated_B + pred_data
            d_fake_B = netD_B(generated_B[:,:,:,rand_int:rand_int+opt.mgcDim])

            # Loss Function
            d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
            d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
            d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

            d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
            d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
            d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

            d_loss = (d_loss_A + d_loss_B) / 2.0

            # Backprop for Generator
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            d_loss.backward()
            optimizerD.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                %(epoch, opt.niter, i, len(data_loader),
                d_loss.item(), g_loss.item()))
            
            if (epoch % 20 == 0) and (epoch != 0):
                
                fake = netG_AB(noise, pred_data)
                fake = fake + pred_data
                fake = fake.data.cpu().numpy()
                fake = fake.reshape(opt.mgcDim, -1)
                fake = fake[:,rand_int:rand_int+opt.mgcDim]
                    
                pred = pred_data.data.cpu().numpy()
                pred = pred.reshape(opt.mgcDim, -1)
                pred = pred[:,rand_int:rand_int+opt.mgcDim]
                    
                real = real_data.cpu().numpy()
                real = real.reshape(opt.mgcDim, -1)
                real = real[:,rand_int:rand_int+opt.mgcDim]
                plot_feats(real, pred, fake,  epoch, i, opt.outf)
            
            del real_data, pred_data, real_data_crop, pred_data_crop,
            del noise, fake_A, fake_A_crop, fake_B, fake_B_crop, generated_A, generated_B 
            torch.cuda.empty_cache()

        # do checkpointing
        if (epoch % 10 == 0) and (epoch != 0):
            torch.save(netG_AB.state_dict(), '%s/netG_AB_epoch_%d.pth' %(opt.outf, epoch))
            # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %(opt.outf, epoch))

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
    parser.add_argument('--batchSize', type=int, default=48, help='input batch size')
    parser.add_argument('--mgcDim', type=int, default=40, help='mel-cepstrum dimension')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--testdata_dir', type=str, help='path to test data')
    opt = parser.parse_args()
    print(opt)
    print(torch.__version__)
    device = torch.device("cuda:0" if opt.cuda else "cpu")
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
    netG_AB = define_netG(in_ch=2, device=device)
    if opt.netG != '':
        netG_AB.load_state_dict(torch.load(opt.netG))
    print(netG_AB)

    netG_BA = define_netG(in_ch=2, device=device)

    # define the discriminator
    netD_A = define_netD(device=device)
    if opt.netD != '':
        netD_A.load_state_dict(torch.load(opt.netD))
    print(netD_A)

    netD_B = define_netD(device=device)

    if opt.mode == 'train':
        train(netD_A=netD_A, netD_B=netD_B, netG_AB=netG_AB, netG_BA=netG_BA, data_loader=data_loader, opt=opt, device=device)
    else:
        print('Mode must be either train or test only')
