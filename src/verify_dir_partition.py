import numpy as np
import json
import torch
from collections import  defaultdict
from pathlib import Path
import os
import copy
from math import *
import random
#from centroidsinit import *
import numpy as np
#from office_dataset import prepare_data
#from centroidutils import *
import argparse
import pickle
import torch.nn as nn
import torchvision.transforms as transforms
#from domainnet_dataset import DomainNetDataset
#from domainnet_dataset_small import DomainNetDatasetSmall
from tiny_imagenet import tinyImageNetDataset

import matplotlib.pyplot as plt
#import torchvision.transforms as transforms
#from torchvision.datasets import CIFAR100

#import pdb; pdb.set_trace()
dataset = 'imagenet'

if dataset == 'cifar100':
    with open(f'dir_0.3_cifar_data.pkl','rb') as f:
        mapping_dict=pickle.load(f)
else:
    with open(f'dir_0.3_imagenet_data.pkl','rb') as f:
        mapping_dict=pickle.load(f)

data_loader_dict=mapping_dict['dataloader']
idx = 75
train_dl_local = data_loader_dict[idx]['train_dl_local']
test_dl_local = data_loader_dict[idx]['test_dl_local']

label_list_train = []
for batch_idx, (x, target,_) in enumerate(train_dl_local):
    label_list_train.append(target.reshape(-1).long())

label_list_train = torch.cat(label_list_train,dim=0)

train_bin_count = torch.bincount(label_list_train)
train_bin_count = train_bin_count/torch.sum(train_bin_count)

#import pdb; pdb.set_trace()
label_list_test = []
test_dl_local = [test_dl_local]
for tmp in test_dl_local:
    for batch_idx, (x, target) in enumerate(tmp):
        label_list_test.append(target.reshape(-1).long())

label_list_test = torch.cat(label_list_test,dim=0)

test_bin_count = torch.bincount(label_list_test)
test_bin_count = test_bin_count/torch.sum(test_bin_count)

plt.plot(train_bin_count.reshape(-1).numpy(),label='train_dist')
plt.plot(test_bin_count.reshape(-1).numpy(),label='test_dist')
plt.legend()

name = dataset + '_dir_dist' + str(idx) + '.png'
folder_name  = 'DirPlots/'
plt.savefig(folder_name+name)
#print("train_bin_count:",train_bin_count)





