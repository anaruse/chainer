import sys
import chainer
from chainer import cuda, Function, gradient_check, utils, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import math

use_GPU = True
#use_GPU = False

if (use_GPU):
    import cupy as xp
else:
    import numpy as xp

import cupy.cuda.nvtx as nvtx
import cupy.cuda.stream as stream
import cupy.cuda.runtime as runtime

# set workspace size for cuDNN
_free_mem, total_mem = cuda.cupy.cuda.runtime.memGetInfo()
size = long(total_mem * 0.1)
cuda.set_max_workspace_size(size)

############################################################

class VGG16(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(  3,  64, 3, pad=1, use_cudnn=use_cudnn),
            conv1_2=L.Convolution2D( 64,  64, 3, pad=1, use_cudnn=use_cudnn),
            conv2_1=L.Convolution2D( 64, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv3_1=L.Convolution2D(128, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv4_1=L.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_1=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, x, t):
        h1_1 = F.relu(self.conv1_1(x   ), use_cudnn=self.use_cudnn)
        h1_2 = F.relu(self.conv1_2(h1_1), use_cudnn=self.use_cudnn)
        h1   = F.max_pooling_2d(h1_2, 2, stride=2, use_cudnn=self.use_cudnn)
        h2_1 = F.relu(self.conv2_1(h1  ), use_cudnn=self.use_cudnn)
        h2_2 = F.relu(self.conv2_2(h2_1), use_cudnn=self.use_cudnn)
        h2   = F.max_pooling_2d(h2_2, 2, stride=2, use_cudnn=self.use_cudnn)
        h3_1 = F.relu(self.conv3_1(h2  ), use_cudnn=self.use_cudnn)
        h3_2 = F.relu(self.conv3_2(h3_1), use_cudnn=self.use_cudnn)
        h3_3 = F.relu(self.conv3_3(h3_2), use_cudnn=self.use_cudnn)
        h3   = F.max_pooling_2d(h3_3, 2, stride=2, use_cudnn=self.use_cudnn)
        h4_1 = F.relu(self.conv4_1(h3  ), use_cudnn=self.use_cudnn)
        h4_2 = F.relu(self.conv4_2(h4_1), use_cudnn=self.use_cudnn)
        h4_3 = F.relu(self.conv4_3(h4_2), use_cudnn=self.use_cudnn)
        h4   = F.max_pooling_2d(h4_3, 2, stride=2, use_cudnn=self.use_cudnn)
        h5_1 = F.relu(self.conv5_1(h4  ), use_cudnn=self.use_cudnn)
        h5_2 = F.relu(self.conv5_2(h5_1), use_cudnn=self.use_cudnn)
        h5_3 = F.relu(self.conv5_3(h5_2), use_cudnn=self.use_cudnn)
        h5   = F.max_pooling_2d(h5_3, 2, stride=2, use_cudnn=self.use_cudnn)
        h6   = F.relu(self.fc6(h5), use_cudnn=self.use_cudnn)
        h7   = F.relu(self.fc7(h6), use_cudnn=self.use_cudnn)
        h8   = self.fc8(h7)
        loss = F.softmax_cross_entropy(h8, t)
        return loss

############################################################

class VGG16_11(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_11, self).__init__(
            conv1_1=L.Convolution2D(  3,  64, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        h1_1 = F.relu(self.conv1_1(x   ), use_cudnn=self.use_cudnn)
        return h1_1

class VGG16_12(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_12, self).__init__(
            conv1_2=L.Convolution2D( 64,  64, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h1_1):
        h1_2 = F.relu(self.conv1_2(h1_1), use_cudnn=self.use_cudnn)
        h1   = F.max_pooling_2d(h1_2, 2, stride=2, use_cudnn=self.use_cudnn)
        return h1

class VGG16_21(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_21, self).__init__(
            conv2_1=L.Convolution2D( 64, 128, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h1):
        h2_1 = F.relu(self.conv2_1(h1  ), use_cudnn=self.use_cudnn)
        return h2_1

class VGG16_22(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_22, self).__init__(
            conv2_2=L.Convolution2D(128, 128, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h2_1):
        h2_2 = F.relu(self.conv2_2(h2_1), use_cudnn=self.use_cudnn)
        h2   = F.max_pooling_2d(h2_2, 2, stride=2, use_cudnn=self.use_cudnn)
        return h2

class VGG16_31(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_31, self).__init__(
            conv3_1=L.Convolution2D(128, 256, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h2):
        h3_1 = F.relu(self.conv3_1(h2  ), use_cudnn=self.use_cudnn)
        return h3_1

class VGG16_32(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_32, self).__init__(
            conv3_2=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h3_1):
        h3_2 = F.relu(self.conv3_2(h3_1), use_cudnn=self.use_cudnn)
        return h3_2

class VGG16_33(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_33, self).__init__(
            conv3_3=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h3_2):
        h3_3 = F.relu(self.conv3_3(h3_2), use_cudnn=self.use_cudnn)
        h3   = F.max_pooling_2d(h3_3, 2, stride=2, use_cudnn=self.use_cudnn)
        return h3

class VGG16_41(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_41, self).__init__(
            conv4_1=L.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h3):
        h4_1 = F.relu(self.conv4_1(h3  ), use_cudnn=self.use_cudnn)
        return h4_1

class VGG16_42(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_42, self).__init__(
            conv4_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h4_1):
        h4_2 = F.relu(self.conv4_2(h4_1), use_cudnn=self.use_cudnn)
        return h4_2

class VGG16_43(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_43, self).__init__(
            conv4_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h4_2):
        h4_3 = F.relu(self.conv4_3(h4_2), use_cudnn=self.use_cudnn)
        h4   = F.max_pooling_2d(h4_3, 2, stride=2, use_cudnn=self.use_cudnn)
        return h4

class VGG16_51(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_51, self).__init__(
            conv5_1=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h4):
        h5_1 = F.relu(self.conv5_1(h4  ), use_cudnn=self.use_cudnn)
        return h5_1

class VGG16_52(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_52, self).__init__(
            conv5_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h5_1):
        h5_2 = F.relu(self.conv5_2(h5_1), use_cudnn=self.use_cudnn)
        return h5_2

class VGG16_53(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_53, self).__init__(
            conv5_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h5_2):
        h5_3 = F.relu(self.conv5_3(h5_2), use_cudnn=self.use_cudnn)
        h5   = F.max_pooling_2d(h5_3, 2, stride=2, use_cudnn=self.use_cudnn)
        return h5

class VGG16_61(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_61, self).__init__(
            fc6=L.Linear(25088, 4096),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h5):
        h6   = F.relu(self.fc6(h5), use_cudnn=self.use_cudnn)
        return h6

class VGG16_71(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_71, self).__init__(
            fc7=L.Linear(4096, 4096),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h6):
        h7   = F.relu(self.fc7(h6), use_cudnn=self.use_cudnn)
        return h7

class VGG16_81(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_81, self).__init__(
            fc8=L.Linear(4096, 1000),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, h7, t):
        h8   = self.fc8(h7)
        loss = F.softmax_cross_entropy(h8, t)
        return loss

############################################################

class Pipeline(object):
    def __init__(self, class_optimizer):
        self.optimizer = class_optimizer
        self.models = []
        self.opts = []
        self.streams = []
        # self.streams.append( None )
        # self.streams.append( None )
        self.streams.append( stream.Stream(non_blocking=True) )
        self.streams.append( stream.Stream(non_blocking=True) )
        self.run_count = 0
        self.outputs = []

    def add_model(self, model):
        self.models.append(model)
        opt = self.optimizer()
        opt.setup(model)
        self.opts.append(opt)

    def setup(self):
        num_stage = len(self.models)
        i = 0
        model = self.models[i]
        model.freegrads()
        model.to_gpu()
        for i in range(1, num_stage):
            model = self.models[i]
            model.freegrads()
            model.to_swap()

    def finalize(self):
        num_stage = len(self.models)
        for i in range(0, num_stage):
            model = self.models[i]
            model.to_cpu()

    def run_sync(self, x, t):
        print("Pipeline: run: {}".format(self.run_count), self.run_count)
        nvtx.RangePush("Run: {}".format(self.run_count), self.run_count)
        num_stage = len(self.models)
        x.to_gpu()
        # Foward
        print("Forward")
        nvtx.RangePush("Pipeline: Forward", 1)
        next_input = x
        for i in range(0, num_stage):
            print("Pipeline: Forward: Stage:{} start".format(i))
            nvtx.RangePush("Layer: {}".format(i), i)

            cur_input = next_input
            cur_model = self.models[i]
            cur_model.to_gpu()

            if i < num_stage-1:
                cur_output = cur_model(cur_input)
            else:
                cur_output = cur_model(cur_input, t)
            cur_output.interrupt_backward()
            cur_output.fores_to_swap()
            self.outputs.append(cur_output)

            next_input = cur_output
            cur_output = None

            print("Pipeline: Forward: Stage:{} end".format(i))
            nvtx.RangePop()

        next_input = None
        nvtx.RangePop()

        # Backward & Update
        print("Backward & Update")
        nvtx.RangePush("Backward & Update", 2)
        for i in reversed(range(0, num_stage)):

            print("Pipeline: Backward & Update: Stage:{} start".format(i))
            nvtx.RangePush("Layer: {}".format(i), i)

            cur_output = self.outputs.pop()
            cur_output.fores_to_gpu()
            cur_output.resume_backward()

            cur_model = self.models[i]
            cur_model.zerograds()
            cur_output.backward()
            cur_opt = self.opts[i]
            cur_opt.update()
            print("unchain_backward: start")
            print sys.getrefcount(cur_output)
            cur_output.unchain_backward() # ...
            print sys.getrefcount(cur_output)
            cur_output = None # ...
            print("unchain_backward: end")
            # cur_model.freegrads()  # ...

            cur_model.to_swap()

            print("Pipeline: Backward & Update: Stage:{} end".format(i))
            nvtx.RangePop()

        nvtx.RangePop()
        #
        self.run_count += 1
        nvtx.RangePop()

    def run(self, x, t):
        print("Pipeline: run: {}".format(self.run_count), self.run_count)
        nvtx.RangePush("Run: {}".format(self.run_count), self.run_count)
        num_stage = len(self.models)
        x.to_gpu()
        # swap-in data of 1st stage (just in case)
        if True:
            stream = self.streams[0]
            next_model = self.models[0]
            next_model.freegrads()  # to reduce amount of data transfer
            next_model.to_gpu(stream=stream)

        # Foward
        print("Pipeline: Forward")
        nvtx.RangePush("Forward", 1)
        next_input = x
        for i in range(0, num_stage):
            # swap-in data of next stage
            print("swap-in: start")
            stream = self.streams[0]
            if stream is not None:
                print("stream.synchronize() start")
                stream.synchronize()
                print("stream.synchronize() end")
            if i < num_stage-1:
                next_model = self.models[i+1]
                next_model.freegrads()  # to reduce amount of data transfer
                next_model.to_gpu(stream=stream)
            print("swap-in: end")

            # do forward computation of current stage
            print("Pipeline: Forward: Stage:{} start".format(i))
            nvtx.RangePush("Layer: {}".format(i), i)
            cur_model = self.models[i]
            cur_input = next_input
            if i < num_stage-1:
                cur_output = cur_model(cur_input)
            else:
                cur_output = cur_model(cur_input, t)
            runtime.deviceSynchronize()
            cur_output.interrupt_backward()
            print("Pipeline: Forward: Stage:{} end".format(i))
            nvtx.RangePop()

            # swap-out data of current stage
            print("swap-out: start")
            stream = self.streams[1]
            if stream is not None:
                print("stream.synchronize() start")
                stream.synchronize()
                print("stream.synchronize() end")
            if i < num_stage-1:
                cur_output.fores_to_swap(stream=stream)

            self.outputs.append(cur_output)
            next_input = cur_output
            print("swap-out: end")

        next_input = None
        nvtx.RangePop()

        # Backward & Update
        print("Pipeline: Backward & Update")
        nvtx.RangePush("Backward & Update", 2)
        for i in reversed(range(0, num_stage)):
            # swap-in data of next stage
            print("swap-in: start")
            stream = self.streams[0]
            if stream is not None:
                stream.synchronize()
            if i > 0:
                next_output = self.outputs[i-1]
                next_output.fores_to_gpu(stream=stream)
            print("swap-in: end")

            # do backward computation of current stage
            print("Pipeline: Backward & Update: Stage:{} start".format(i))
            nvtx.RangePush("Layer: {}".format(i), i)
            cur_output = self.outputs.pop()
            cur_model = self.models[i]
            cur_opt = self.opts[i]
            cur_model.zerograds()
            cur_output.resume_backward()
            cur_output.backward()
            cur_opt.update()
            runtime.deviceSynchronize()  # ...
            cur_output.unchain_backward() # ...
            cur_output = None # ...
            cur_model.freegrads()  # ...
            print("Pipeline: Backward & Update: Stage:{} end".format(i))
            nvtx.RangePop()

            # swap-out data of current stage
            print("swap-out: start")
            stream = self.streams[1]
            if stream is not None:
                print("stream.synchronize() start")
                stream.synchronize()
                print("stream.synchronize() end")
            if i > 0:
                cur_model.to_swap(stream=stream)
            print("swap-out: end")

        nvtx.RangePop()
        #
        self.run_count += 1
        nvtx.RangePop()

############################################################

def compare_array(name, x, y):
    assert (x.ndim == y.ndim)
    print('# compare\t{}'.format(name))
    if (x.ndim == 1):
        (i_max, ) = x.shape
        for i in range(0,i_max-1):
            if (x[i] == y[i]):
                continue
            if (math.isnan(x[i]) and math.isnan(y[i])):
                continue
            print('\t{}: MMF in ({}): {}, {}'.format(name,i, x[i], y[i]))
    elif (x.ndim == 2):
        (i_max, j_max, ) = x.shape
        for j in range(0,j_max-1):
            for i in range(0,i_max-1):
                if (x[i][j] == y[i][j]):
                    continue
                if (math.isnan(x[i][j]) and math.isnan(y[i][j])):
                    continue
                print('\t{}: MMF in ({},{}): {}, {}'.format(name,i,j,x[i][j],y[i][j]))
    elif (x.ndim == 3):
        (i_max, j_max, k_max, ) = x.shape
        for k in range(0,k_max-1):
            for j in range(0,j_max-1):
                for i in range(0,i_max-1):
                    if (x[i][j][k] == y[i][j][k]):
                        continue
                    if (math.isnan(x[i][j][k]) and math.isnan(y[i][j][k])):
                        continue
                    print('\t{}: MMF in ({},{},{}): {}, {}'.format(name,i,j,k,x[i][j][k],y[i][j][k]))
    elif (x.ndim == 4):
        (i_max, j_max, k_max, l_max, ) = x.shape
        for l in range(0,l_max-1):
            for k in range(0,k_max-1):
                for j in range(0,j_max-1):
                    for i in range(0,i_max-1):
                        if (x[i][j][k][l] == y[i][j][k][l]):
                            continue
                        if (math.isnan(x[i][j][k][l]) and math.isnan(y[i][j][k][l])):
                            continue
                        print('\t{}: MMF in ({},{},{},{}): {}, {}'.format(name,i,j,k,l,x[i][j][k][l],y[i][j][k][l]))
    else:
        raise

def compare_link(name, l1, l2):
    compare_array( name+'.W.data', l1.W.data, l2.W.data )
    compare_array( name+'.b.data', l1.b.data, l2.b.data )
    # compare_array( name+'.W.grad', l1.W.grad, l2.W.grad )
    # compare_array( name+'.b.grad', l1.b.grad, l2.b.grad )

def compare_links():
    compare_link( 'conv1', model1.conv1, model_st1.conv1 )
    compare_link( 'conv2', model1.conv2, model_st2.conv2 )
    compare_link( 'conv3', model1.conv3, model_st3.conv3 )
    compare_link( 'conv4', model1.conv4, model_st4.conv4 )
    compare_link( 'conv5', model1.conv5, model_st5.conv5 )
    compare_link( 'fc6', model1.fc6, model_st6.fc6 )
    compare_link( 'fc7', model1.fc7, model_st7.fc7 )
    compare_link( 'fc8', model1.fc8, model_st8.fc8 )

############################################################

model1 = VGG16(use_cudnn=True)
opt1 = optimizers.SGD()
opt1.setup(model1)

############################################################

model_11 = VGG16_11(use_cudnn=True)
model_12 = VGG16_12(use_cudnn=True)
model_21 = VGG16_21(use_cudnn=True)
model_22 = VGG16_22(use_cudnn=True)
model_31 = VGG16_31(use_cudnn=True)
model_32 = VGG16_32(use_cudnn=True)
model_33 = VGG16_33(use_cudnn=True)
model_41 = VGG16_41(use_cudnn=True)
model_42 = VGG16_42(use_cudnn=True)
model_43 = VGG16_43(use_cudnn=True)
model_51 = VGG16_51(use_cudnn=True)
model_52 = VGG16_52(use_cudnn=True)
model_53 = VGG16_53(use_cudnn=True)
model_61 = VGG16_61(use_cudnn=True)
model_71 = VGG16_71(use_cudnn=True)
model_81 = VGG16_81(use_cudnn=True)

pipeline = Pipeline(optimizers.SGD)
pipeline.add_model( model_11 )
pipeline.add_model( model_12 )
pipeline.add_model( model_21 )
pipeline.add_model( model_22 )
pipeline.add_model( model_31 )
pipeline.add_model( model_32 )
pipeline.add_model( model_33 )
pipeline.add_model( model_41 )
pipeline.add_model( model_42 )
pipeline.add_model( model_43 )
pipeline.add_model( model_51 )
pipeline.add_model( model_52 )
pipeline.add_model( model_53 )
pipeline.add_model( model_61 )
pipeline.add_model( model_71 )
pipeline.add_model( model_81 )

############################################################

model_11.conv1_1.W.copydata( model1.conv1_1.W )
model_11.conv1_1.b.copydata( model1.conv1_1.b )
model_12.conv1_2.W.copydata( model1.conv1_2.W )
model_12.conv1_2.b.copydata( model1.conv1_2.b )

model_21.conv2_1.W.copydata( model1.conv2_1.W )
model_21.conv2_1.b.copydata( model1.conv2_1.b )
model_22.conv2_2.W.copydata( model1.conv2_2.W )
model_22.conv2_2.b.copydata( model1.conv2_2.b )

model_31.conv3_1.W.copydata( model1.conv3_1.W )
model_31.conv3_1.b.copydata( model1.conv3_1.b )
model_32.conv3_2.W.copydata( model1.conv3_2.W )
model_32.conv3_2.b.copydata( model1.conv3_2.b )
model_33.conv3_3.W.copydata( model1.conv3_3.W )
model_33.conv3_3.b.copydata( model1.conv3_3.b )

model_41.conv4_1.W.copydata( model1.conv4_1.W )
model_41.conv4_1.b.copydata( model1.conv4_1.b )
model_42.conv4_2.W.copydata( model1.conv4_2.W )
model_42.conv4_2.b.copydata( model1.conv4_2.b )
model_43.conv4_3.W.copydata( model1.conv4_3.W )
model_43.conv4_3.b.copydata( model1.conv4_3.b )

model_51.conv5_1.W.copydata( model1.conv5_1.W )
model_51.conv5_1.b.copydata( model1.conv5_1.b )
model_52.conv5_2.W.copydata( model1.conv5_2.W )
model_52.conv5_2.b.copydata( model1.conv5_2.b )
model_53.conv5_3.W.copydata( model1.conv5_3.W )
model_53.conv5_3.b.copydata( model1.conv5_3.b )

model_61.fc6.W.copydata( model1.fc6.W )
model_61.fc6.b.copydata( model1.fc6.b )
model_71.fc7.W.copydata( model1.fc7.W )
model_71.fc7.b.copydata( model1.fc7.b )
model_81.fc8.W.copydata( model1.fc8.W )
model_81.fc8.b.copydata( model1.fc8.b )

############################################################

# nbatch=16*1
nbatch=16*2  # unified: OK
# nbatch=16*3
# nbatch=16*4
# nbatch=16*5
# nbatch=16*6
# nbatch=16*7
# nbatch=16*8

x = xp.random.uniform(-1, 1, (nbatch, 3, 224, 224)).astype(xp.float32)
x = Variable( xp.asarray(x) )

x1 = Variable( xp.zeros_like( x.data ) )
x2 = Variable( xp.zeros_like( x.data ) )
x1.copydata(x)
x2.copydata(x)
x1.name = "x1"
x2.name = "x2"
#x1.zerograd()
#x2.zerograd()

label = xp.zeros((nbatch), dtype=xp.int32)
for i in range(0, len(label)):
    label[i] = i % 1000
label = Variable( xp.asarray(label) )

print(label.data)

############################################################

num_loop = 3

############################################################

if True:
    print '#################### unified model ####################'
    if (use_GPU):
        model1.to_gpu()
    
    for loop in range(0, num_loop):
    
        runtime.deviceSynchronize()
        nvtx.RangePush("Run: {}".format(loop), loop)
    
        nvtx.RangePush("Forward",1)
        loss1 = model1(x1, label)
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        print 'loop:{}, loss:{}'.format(loop, loss1.data)
    
        nvtx.RangePush("Backward & Update",2)
        model1.zerograds()
        loss1.backward()
        opt1.update()
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
    if (use_GPU):
        model1.to_cpu()

############################################################

if True:
    print '#################### pipeline model ####################'
    pipeline.setup()
    
    for loop in range(0, num_loop): # 10 loops
        pipeline.run(x2, label)
        # pipeline.run_sync(x2, label)
    
    pipeline.finalize()

############################################################

print('########## check variables ##########')

# compare_links()

# compare_array('y1 and y2', y1.data, y2.data)
# compare_array('x1 and x2', x1.grad, x2.grad)

