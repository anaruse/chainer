import sys
import chainer
from chainer import cuda, Function, gradient_check, utils, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import math

import numpy

use_GPU = True
#use_GPU = False

if (use_GPU):
    import cupy as xp
else:
    import numpy as xp

import cupy.cuda.nvtx as nvtx
import cupy.cuda.stream as stream
import cupy.cuda.runtime as runtime

from time import sleep

import googlenet
import googlenet_ooc

# set workspace size for cuDNN
_free_mem, total_mem = cuda.cupy.cuda.runtime.memGetInfo()
# size = long(total_mem * 0.1)
# size = long(total_mem * 0.01)
size = int(total_mem * 0.01)
cuda.set_max_workspace_size(size)

############################################################

#model0 = googlenet.GoogLeNet()
model0 = googlenet_ooc.GoogLeNet_ooc()
opt0 = optimizers.SGD()
opt0.setup(model0)

############################################################

import argparse
parser = argparse.ArgumentParser(description='example ...')
parser.add_argument('--batch', '-b', type=int, default=16, help='minibatch size')

args = parser.parse_args()
nbatch=args.batch
print('nbatch:{}'.format(nbatch))

############################################################

x = xp.random.uniform(-1, 1, (nbatch, 3, 224, 224)).astype(xp.float32)
x = Variable( xp.asarray(x) )

x0 = Variable( xp.zeros_like( x.data ) )
x0.copydata(x)
x0.name = "x0"
x0.cleargrad()

label = xp.zeros((nbatch), dtype=xp.int32)
for i in range(0, len(label)):
    label[i] = i % 1000
label = Variable( xp.asarray(label) )

############################################################

num_loop = 5

############################################################

if True:
    print('#################### model ####################')
    nvtx.RangePush("Model", 0)
    if (use_GPU):
        model0.to_gpu()
    
    sleep(1)

    for loop in range(0, num_loop):
        runtime.deviceSynchronize()
        nvtx.RangePush("Run: {}".format(loop), loop)
    
        model0.cleargrads()
        nvtx.RangePush("Forward",1)
        loss0 = model0(x0, label)
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        print('loop:{}, loss:{}'.format(loop, loss0.data))
    
        nvtx.RangePush("Backward & Update",2)
        loss0.backward()
        opt0.update()
        runtime.deviceSynchronize()
        nvtx.RangePop()

        runtime.deviceSynchronize()
        nvtx.RangePop()

        sleep(1)
    
    if (use_GPU):
        model0.to_cpu()
    nvtx.RangePop()

############################################################
