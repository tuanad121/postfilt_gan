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
from torch.autograd import grad as torch_grad

from data_loader import get_loader, prepare_normalizer
from utils import plot_feats, read_binary_file
from models_wgan import define_netD, define_netG

num_steps = 0
gp_weight = 10
critic_iterations = 5

def calc_gradient_penalty(netD, real_data, fake_data):
    # print('----------', real_data.size(), fake_data.size())
    alpha = torch.FloatTensor(real_data.size(0),1,1,1).uniform_(0,1)
    alpha = alpha.expand(real_data.size(0), real_data.size(1), real_data.size(2), real_data.size(3))
    
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch_grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # LAMBDA
    return gradient_penalty


def train(netD, netG, data_loader, opt):
    label = torch.FloatTensor(1)
    label = Variable(label, requires_grad=False)
    real_label = 1
    fake_label = -1

    # cost criterion
    # criterion = nn.BCELoss() # normal gan 
    # criterion = nn.MSELoss() # lsgan

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        # criterion.cuda()
    
    one = torch.FloatTensor([1])
    mone = one * -1
    if opt.cuda:
        one = one.cuda()
        mone = mone.cuda()
    
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print('batch size =', opt.batchSize)
    for epoch in range(opt.niter):      

        for i, (real_data, pred_data) in enumerate(data_loader):
            # Discriminator update
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            
            for _ in range(5):
                # clear the gradient buffers
                netD.zero_grad()

                # crop the tensor to fixed size
                rand_int = random.randint(0,real_data.size(-1) - (opt.mgcDim-1))
                real_data_crop = real_data[:,:,:,rand_int:rand_int+opt.mgcDim-1]
                noise = torch.FloatTensor(real_data.size()).normal_(0,1)
                if opt.cuda:
                    pred_data = pred_data.cuda()
                    real_data_crop = real_data_crop.cuda()
                    noise = noise.cuda()                
                pred_data = Variable(pred_data)
                real_crop = Variable(real_data_crop)
                
                # train with real
                d_real = netD(real_crop)
                d_real = d_real.mean()
                d_real.backward(mone)

                # train with fake 
                fake = netG(noise, pred_data) + pred_data
                fake_data_crop = fake[:,:,:,rand_int:rand_int+opt.mgcDim-1]
                fake_crop = Variable(fake_data_crop)
                d_generated = netD(fake_crop)
                d_generated = d_generated.mean()
                d_generated.backward(one)

                gradient_penalty = calc_gradient_penalty(netD, real_crop, fake_crop)
                gradient_penalty.backward()

                d_loss = d_generated - d_real + gradient_penalty
                optimizerD.step()

            
            # Generator update
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            
            netG.zero_grad()
            rand_int = random.randint(0,real_data.size(-1) - (opt.mgcDim-1))
            real_data_crop = real_data[:,:,:,rand_int:rand_int+opt.mgcDim-1]
            noise = torch.FloatTensor(real_data.size()).normal_(0,1)
            if opt.cuda:
                pred_data = pred_data.cuda()
                real_data_crop = real_data_crop.cuda()
                noise = noise.cuda()                
            pred_data = Variable(pred_data, requires_grad=True)
            real_crop = Variable(real_data_crop, requires_grad=True)

            fake = netG(noise, pred_data) + pred_data
            fake_data_crop = fake[:,:,:,rand_int:rand_int+opt.mgcDim-1]
            fake_crop = Variable(fake_data_crop, requires_grad=True)
            d_generated = netD(fake_crop)
            d_generated = d_generated.mean()
            d_generated.backward(mone)
            # if 0:
            #     errRes = nn.MSELoss()(fake, real_data)
            #     g_loss = errRes + errG
            # else:
            #     g_loss = errG
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f D(G(z)): %.4f'
                %(epoch, opt.niter, i, len(data_loader),
                d_loss, d_generated.item()))
            
            
            del real_data, pred_data, 
            del noise, real_data_crop, fake, fake_crop
            torch.cuda.empty_cache()

        # do checkpointing
        if (epoch % 20 == 0) and (epoch != 0):
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %(opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %(opt.outf, epoch))

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
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--mgcDim', type=int, default=40, help='mel-cepstrum dimension')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
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
    netG = define_netG(in_ch=2, device=device)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # define the discriminator
    netD = define_netD(device=device)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    if opt.mode == 'train':
        train(netD, netG, data_loader, opt)
    elif opt.mode == 'test':
        test(netG, opt)
    else:
        print('Mode must be either train or test only')
