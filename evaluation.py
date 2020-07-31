from __future__ import print_function
import argparse
import os
import gc
import sys
import xlwt
import random
import numpy as np
from advertorch.attacks import LinfBasicIterativeAttack, CarliniWagnerL2Attack
from advertorch.attacks import GradientSignAttack, PGDAttack
import foolbox
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp

from net import Net_s, Net_m, Net_l
SEED = 10000
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(10000)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading\
    workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--adv', type=str, help='attack method')
parser.add_argument('--mode', type=str, help='use which model to generate\
    examples. "imitation_large": the large imitation network.\
    "imitation_medium": the medium imitation network. "imitation_small" the\
    small imitation network. ')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--target', action='store_true', help='manual seed')

opt = parser.parse_args()
# print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \
         --cuda")

testset = torchvision.datasets.MNIST(root='/data/dataset/', train=False,
                                     download=True,
                                     transform=transforms.Compose([
                                        transforms.ToTensor(),
                                     ]))

data_list = [i for i in range(0, 10000)]
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=2)


device = torch.device("cuda:0" if opt.cuda else "cpu")

# L2 = foolbox.distances.MeanAbsoluteDistance()

def test_adver(net, tar_net, attack, target):
    net.eval()
    tar_net.eval()
    # BIM
    if attack == 'BIM':
        adversary = LinfBasicIterativeAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.25,
            nb_iter=120, eps_iter=0.02, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
    # PGD
    elif attack == 'PGD':
        if opt.target:
            adversary = PGDAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=0.25,
                nb_iter=11, eps_iter=0.03, clip_min=0.0, clip_max=1.0,
                targeted=opt.target)
        else:
            adversary = PGDAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=0.25,
                nb_iter=6, eps_iter=0.03, clip_min=0.0, clip_max=1.0,
                targeted=opt.target)
    # FGSM
    elif attack == 'FGSM':
        adversary = GradientSignAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.26,
            targeted=opt.target)
    elif attack == 'CW':
        adversary = CarliniWagnerL2Attack(
            net,
            num_classes=10,
            learning_rate=0.45,
            # loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            binary_search_steps=10,
            max_iterations=12,
            targeted=opt.target)

    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        net.eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
        print('Accuracy of the network on netD: %.2f %%' %
                (100. * correct_netD.float() / total))

    # correct = 0.0
    # total = 0.0
    # tar_net.eval()
    # total_L2_distance = 0.0
    # for data in testloader:
    #     inputs, labels = data
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)
    #     outputs = tar_net(inputs)
    #     _, predicted = torch.max(outputs.data, 1)
    #     if target:
    #         # randomly choose the specific label of targeted attack
    #         labels = torch.randint(0, 9, (1,)).to(device)
    #         # test the images which are not classified as the specific label
    #         if predicted != labels:
    #             # print(total)
    #             adv_inputs_ori = adversary.perturb(inputs, labels)
    #             L2_distance = (torch.norm(adv_inputs_ori - inputs)).item()
    #             total_L2_distance += L2_distance
    #             with torch.no_grad():
    #                 outputs = tar_net(adv_inputs_ori)
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 total += labels.size(0)
    #                 correct += (predicted == labels).sum()
    #     else:
    #         # test the images which are classified correctly
    #         if predicted == labels:
    #             # print(total)
    #             adv_inputs_ori = adversary.perturb(inputs, labels)
    #             L2_distance = (torch.norm(adv_inputs_ori - inputs)).item()
    #             total_L2_distance += L2_distance
    #             with torch.no_grad():
    #                 outputs = tar_net(adv_inputs_ori)
    #                 _, predicted = torch.max(outputs.data, 1)

    #                 total += labels.size(0)
    #                 correct += (predicted == labels).sum()
    #                 if total < 100:
    #                     vutils.save_image(adv_inputs_ori.detach(),
    #                             'output/' + opt.adv + '_' + str(total) + '.png',
    #                             normalize=False)
            # adv_inputs_ori = adversary.perturb(inputs, labels)
            # L2_distance = (torch.norm(adv_inputs_ori - inputs) / torch.numel(adv_inputs_ori)).item()
            # total_L2_distance += L2_distance
            # with torch.no_grad():
            #     outputs = tar_net(adv_inputs_ori)

    # if target:
    #     print('Attack success rate: %.2f %%' %
    #           (100. * correct.float() / total))
    # else:
    #     print('Attack success rate: %.2f %%' %
    #           (100.0 - 100. * correct.float() / total))
    # print('l2 distance:  %.4f ' % (total_L2_distance / total))


target_net = Net_m().to(device)
state_dict = torch.load(
    'pretrained/net_m.pth')
target_net.load_state_dict(state_dict)
target_net.eval()

if opt.mode == 'black':
    attack_net = Net_l().to(device)
    state_dict = torch.load(
        'pretrained/net_l.pth')
    attack_net.load_state_dict(state_dict)
elif opt.mode == 'white':
    attack_net = target_net
elif opt.mode == 'dast':
    attack_net = Net_l().to(device)
    state_dict = torch.load(
        'saved_model_2/netD_epoch_670.pth')
    attack_net = nn.DataParallel(attack_net)
    attack_net.load_state_dict(state_dict)

test_adver(attack_net, target_net, opt.adv, opt.target)