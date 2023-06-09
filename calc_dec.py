import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import os

n_conv = 18
names = pd.read_csv('resnet{}_layers.csv'.format(n_conv))

path = 'TIN/resnet{}/results/'.format(n_conv)

save = 'hflip_rcrop'

none1_mats = os.listdir(path + 'none1')
none1_mats.remove('none1_none1.tf')
none1_mats.remove('none2_none1.tf')
none2_mats = os.listdir(path + 'none2')
none2_mats.remove('none2_none2.tf')
none2_mats.remove('none1_none2.tf')

none_mat = torch.load(path + 'none1/none2_none1.tf').numpy()

aug_mat1 = torch.load(path + 'none1/{}_none1.tf'.format(save))


x = [i for i in range(1, n_conv)]

conv_layers = []

# get indecies of convolutional layers
for i in range(len(names.values)):
    
    if ("Conv" in names.values[i][1] and "conv" not in names.values[i][1] and "BatchNorm2d" not in names.values[i][1]):
        conv_layers.append(i + 1)        


none_cka_1 = []
aug = []

for i in conv_layers:
    none_cka_1.append(none_mat[i, i])
    aug.append(aug_mat[i, i])

diff = np.divide((np.array(none_cka_1) - np.array(aug)), np.array(none_cka_1))
diff *= 100

df = pd.DataFrame(diff)
df.to_csv('lines/' + save + '/{}_scratch.csv'.format(n_conv), index=False)


########################### pre-trained ##############################

path = 'TIN/resnet{}_pre/results/'.format(n_conv)

none1_mats = os.listdir(path + 'none1')
none1_mats.remove('none1_none1.tf')
none1_mats.remove('none2_none1.tf')
none2_mats = os.listdir(path + 'none2')
none2_mats.remove('none2_none2.tf')
none2_mats.remove('none1_none2.tf')

none_mat = torch.load(path + 'none1/none2_none1.tf').numpy()

aug_mat1 = torch.load(path + 'none1/{}_none1.tf'.format(save))
aug_mat2 = torch.load(path + 'none2/{}_none2.tf'.format(save))
aug_mat = (aug_mat1.add(aug_mat2)) / 2


x = [i for i in range(1, n_conv)]

conv_layers = []

# get indecies of convolutional layers
for i in range(len(names.values)):
    if ("Conv" in names.values[i][1] and "conv" not in names.values[i][1] and "BatchNorm2d" not in names.values[i][1]):
        conv_layers.append(i)  # After activation function
        #print(names.values[i][1])
        


none_cka = []
aug = []

for i in conv_layers:
    none_cka.append(none_mat[i, i])
    aug.append(aug_mat[i, i])


diff_pre = np.divide((np.array(none_cka) - np.array(aug)), np.array(none_cka))
diff_pre *= 100

df = pd.DataFrame(diff_pre)
df.to_csv('lines/' + save + '/{}_pre.csv'.format(n_conv), index=False)