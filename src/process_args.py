import argparse
from utils_general import Config

ALLOW_VALUES = {
    "model_name": ["cifar10", "cifar100", "tiny_vit", "vit-small"],
    "dataset_name": ["CIFAR10", "CIFAR100", "CIFAR10C", "TinyImageNet", "IN100", "shakespeare"],
    "alg_name": ["FedAvg", "FedAvgReg", "FedDyn", "FedProx", "FedSpeed"],
    "rule": ["Dirichlet", "iid"]
}

def parse_args():
    args = Config('config.yml')
    parser = argparse.ArgumentParser(description='Experiment configuration')
    parser.add_argument('--device', type=str, help='cuda/cpu')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--use_checkpoint', type=int, help='load from checkpoint')
    parser.add_argument('--restart_round', type=int, help='restart from this round')
    cl_args = parser.parse_args()
    
    for key, value in vars(cl_args).items():
        if value is not None:
            args.key = value
            
    return args

def validate_args(args):
    for key, value in vars(args).items():
        allowed = ALLOW_VALUES[key]
        if isinstance(allowed, list):
            if value not in allowed:
                raise ValueError(f'Invalid value for \'{key}\':{value}. Allowed values are: {allowed}')
    
    print('All arguments are valid.')    