import time

import torch
from torch.nn import functional as func
import numpy
from scipy import stats

from hyperspherical.von_mises_fisher import VonMisesFisher

mu = func.normalize(torch.randn(100), dim=-1)

vmf = VonMisesFisher(mu, 10)
start = time.time()
samples = vmf.sample(torch.Size([10000000]))
print(time.time() - start)
mean = torch.mean(samples, dim=0)

print(mean)
print(vmf.mean)

print(vmf.entropy())

cos1 = torch.mean(samples @ mu)

mu = mu.numpy()
vmf = stats.vonmises_fisher(mu, 10)
start = time.time()
samples = vmf.rvs(size=10000000)
print(time.time() - start)
print(samples.shape)
cos2 = numpy.mean(samples @ mu)
print(cos1, cos2)

print(vmf.entropy())
