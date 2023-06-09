import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import os

path = 'cutmix/'

layers_18 = pd.read_csv(path + '18_scratch.csv')
layers_18_pre = pd.read_csv(path + '18_pre.csv')
layers_50 = pd.read_csv(path + '50_scratch.csv')
layers_50_pre = pd.read_csv(path + '50_pre.csv')
layers_32 = pd.read_csv(path + '32_scratch.csv')
layers_56 = pd.read_csv(path + '56_scratch.csv')

layers_50.iloc[6:11] /= 10

x_18 = [x / 17 for x in range(1, 18)]
x_32 = [x / 31 for x in range(1, 32)]
x_50 = [x / 49 for x in range(1, 50)]
x_56 = [x / 55 for x in range(1, 56)]

plt.style.use('bmh')
#plt.plot(x_18, layers_18, color='red', label = '18 Scratch')
plt.plot(x_18, layers_18_pre, color='crimson', label='R-18')
#plt.plot(x_50, layers_50, color='blue', label = '50 Scratch')
plt.plot(x_50, layers_50_pre, color='mediumblue', label='R-50')
plt.ylim(-10, 60)
plt.title('Solarize')
#plt.legend()

plt.show()

plt.plot(x_32, layers_32, color='forestgreen', label = 'R-32')
plt.plot(x_56, layers_56, color='darkorange', label='R-56')
plt.ylim(-10, 60)
plt.title('Solarize')
#plt.legend()

plt.show()