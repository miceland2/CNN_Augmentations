import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
from resnet import resnet32, resnet56, resnet110
from torchvision.models import resnet50, resnet18
import torchvision.datasets as datasets
from collections import OrderedDict
from torch_cka import CKA
import os


def generate_dataloader(data, name, transform, use_cuda):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=torch.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = torch.utils.data.DataLoader(dataset, 256, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader

def sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub))
    return x.translate(res)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'val/Images')

path = 'C10/resnet32/'
state1 = torch.load(path + 'none1/model.th')
augs = os.listdir(path)
augs.remove('results')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
t_tin = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

t_c = transforms.Compose([
        transforms.ToTensor(),
        normalize
])

batch_size = 32

dataset = datasets.ImageFolder(VALID_DIR, transform=t_tin)

subset = list(range(0, len(dataset), 5))
dataset = torch.utils.data.Subset(dataset, subset)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                        shuffle=False, worker_init_fn=seed_worker,
                        generator=g)

#test_dataset = datasets.CIFAR10(root='../../data/',
#                                            train=False, 
#                                            transform=t_c)

#loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                          batch_size=batch_size, 
#                                          shuffle=False)

for aug in augs:

    state2 = torch.load(path + aug + '/model.th')

    model1 = resnet32()
    model1.fc = nn.Linear(512, 200)
    model2 = resnet32()
    model2.fc = nn.Linear(512, 200)

    state_dict1 = state1['state_dict']
    state_dict2 = state2['state_dict']

    new_state_dict1 = OrderedDict()
    for k, v in state_dict1.items():
        name = k[7:] # remove `module.`
        new_state_dict1[name] = v

    new_state_dict2 = OrderedDict()
    for k, v in state_dict2.items():
        name = k[7:] # remove `module.`
        new_state_dict2[name] = v

    model1.load_state_dict(new_state_dict1, strict=False)
    model1.eval()
    model2.load_state_dict(new_state_dict2, strict=False)
    model2.eval()

    cka = CKA(model1, model2,
             model1_name="none" + sub("1"), model2_name=aug,
             device='cuda')
    cka.compare(loader)

    results = cka.export()

    cka.plot_results(save_path=path + "results/" + aug + ".png")

    mat = results['CKA']
    torch.save(mat, path + 'results/' + aug + '_none2' + '.tf')