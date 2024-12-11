
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import pickle
from collections import defaultdict
import random
import logging
import torch.utils.data as data


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

n_parties=200
batch_size=32
cls_num=10
img_size=224

class tinyImageNetDataset(Dataset):
    def __init__(self, train=True, transform=None, return_index=False, dataidxs=None, base_dir=None):
        self.base_path=base_dir
        self.dirlist=os.listdir(base_dir + ('/train' if train else '/test'))
        if(train):
            self.base_path=self.base_path+"/train"
        else:
            self.base_path=self.base_path+"/test"
        self.image_paths=[]
        self.labels=[]
        for dir in self.dirlist:
            root_path=self.base_path+f"/{dir}"
            imgs=os.listdir(root_path)
            for img in imgs:
                self.image_paths.append(root_path+f"/{img}")
                self.labels.append(self.dirlist.index(dir))
        if(dataidxs is not None):
            self.image_paths=[self.image_paths[i] for i in dataidxs]
            self.labels=[self.labels[i] for i in dataidxs]
        self.labels=np.array(self.labels)
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_paths[idx])
        label = self.labels[idx]
        label=torch.tensor(label,dtype=torch.long)
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)
        if self.return_index:
            return image, label,idx
        else:
            return image, label
        
class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):

        return torch.clamp((tensor + torch.randn(tensor.size()) * self.std + self.mean), 0, 255)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def load_tinyimagenet_data(base_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = tinyImageNetDataset(train=True,transform=transform, base_dir=base_dir)
    test_ds = tinyImageNetDataset(train=False, transform=transform, base_dir=base_dir)

    y_train = train_ds.labels
    y_test = test_ds.labels
    return (y_train, y_test)

def partition_tinyimagenet_data(base_dir):
    y_train, y_test = load_tinyimagenet_data(base_dir)
    num = cls_num
    K = 100
        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
    assert (num * n_parties) % K == 0, "equal classes appearance is needed"
    count_per_class = (num * n_parties) // K
    class_dict = {}
    for i in range(K):
            # sampling alpha_i_c
        probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(n_parties):
        c = []
        for _ in range(num):
            class_counts = [class_dict[i]['count'] for i in range(K)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
    data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
    data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

    num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
    num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
    for data_idx in data_class_idx_train.values():
        random.shuffle(data_idx)
    for data_idx in data_class_idx_test.values():
        random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
    net_dataidx_map_train ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
    net_dataidx_map_test ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
    for usr_i in range(n_parties):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx_train = int(num_samples_train[c] * p)
            end_idx_test = int(num_samples_test[c] * p)
            net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i], data_class_idx_train[c][:end_idx_train])
            net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i], data_class_idx_test[c][:end_idx_test])

            data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
            data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]
    return net_dataidx_map_train, net_dataidx_map_test


def get_dirichlet_partition_tinyimagenet_dataloader(n_client,n_cls=100,rule_arg=0.3,base_dir=None):
    
    cls_priors   = np.random.dirichlet(alpha=[rule_arg]*n_cls,size=n_client)
    prior_cumsum = np.cumsum(cls_priors, axis=1)
    
    y_train, y_test = load_tinyimagenet_data(base_dir)

    n_data_per_clnt =  len(y_train)//n_client
    net_dataidx_map_train = dirichlet_partition(n_client,prior_cumsum,y_train,n_cls,n_data_per_clnt)
    
    n_data_per_clnt = len(y_test)// n_client
    net_dataidx_map_tst  = dirichlet_partition(n_client,prior_cumsum,y_test,n_cls,n_data_per_clnt)

    return net_dataidx_map_train, net_dataidx_map_tst
    
        # --------- #

def dirichlet_partition(n_client,prior_cumsum,data_y,n_cls,n_data_per_clnt):
    #clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(n_client) ]
    net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_client)}
    data_class_idx= {i: np.where(data_y == i)[0] for i in range(n_cls)}
    num_samples = {i: len(data_class_idx[i]) for i in range(n_cls)}
    
    #idx_list = [np.where(data_y==i)[0] for i in range(n_cls)]
    #cls_amount = [len(idx_list[i]) for i in range(n_cls)]
    clnt_data_list = (np.ones(n_client) * n_data_per_clnt).astype(int)
                
    while(np.sum(clnt_data_list)!=0):
        curr_clnt = np.random.randint(n_client)
        # If current node is full resample a client
        print('Remaining Data: %d' %np.sum(clnt_data_list))
        if clnt_data_list[curr_clnt] <= 0:
            continue
        clnt_data_list[curr_clnt] -= 1
        curr_prior = prior_cumsum[curr_clnt]
        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if trn_y is out of that class
            if num_samples[cls_label] <= 0:
                continue
            num_samples[cls_label] -= 1
                                    
            #clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = data_x[data_class_idx[cls_label][num_samples[cls_label]]]
            #clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = data_y[data_class_idx[cls_label][num_samples[cls_label]]]

            net_dataidx_map[curr_clnt] = np.append(net_dataidx_map[curr_clnt], data_class_idx[cls_label][num_samples[cls_label]])
            break

    return net_dataidx_map


def get_tinyimagenet_dataloader(dataidxs_train, dataidxs_test,noise_level=0, drop_last=False, apply_noise=False):
    train_bs = batch_size
    test_bs =  batch_size
    dl_obj=tinyImageNetDataset
    if apply_noise:
        transform_train = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            GaussianNoise(0., noise_level)
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            GaussianNoise(0., noise_level)
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    train_ds = dl_obj(train=True, transform=transform_train,return_index=True,dataidxs=dataidxs_train )
    test_ds = dl_obj(train=False, transform=transform_test,return_index=False,dataidxs= dataidxs_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl

partition = 'Dirichlet'
arr=np.arange(n_parties)
if partition == 'pl':
   net_dataidx_map_train, net_dataidx_map_test = partition_tinyimagenet_data(base_dir=sys.argv[1])
elif partition == 'Dirichlet':
   net_dataidx_map_train, net_dataidx_map_test = get_dirichlet_partition_tinyimagenet_dataloader(200, rule_arg=0.3, base_dir=sys.argv[1])

data_loader_dict = {}
for net_id in arr:
   dataidxs_train = net_dataidx_map_train[net_id]
   dataidxs_test = net_dataidx_map_test[net_id]
   data_loader_dict[net_id] = {}
   train_dl_local, test_dl_local = get_tinyimagenet_dataloader(dataidxs_train, dataidxs_test)
   data_loader_dict[net_id]['train_dl_local'] = train_dl_local
   data_loader_dict[net_id]['test_dl_local'] = test_dl_local

mapping_dict={}
mapping_dict['dataloader']=data_loader_dict
mapping_dict['train']=net_dataidx_map_train

with open("dir_0.3_imagenet_data.pkl","wb") as f:
   pickle.dump(mapping_dict,f)