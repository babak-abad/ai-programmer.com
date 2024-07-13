import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
import util as utl
import configs as cfg

data = [cfg.step_x*x for x in range(0, int(10*np.pi/cfg.step_x))]

data = (np.sin(data) + 1)/2.0

plt.plot(data)
plt.show()