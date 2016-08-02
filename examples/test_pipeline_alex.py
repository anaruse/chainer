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
size = long(total_mem * 0.01)
cuda.set_max_workspace_size(size)


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
        nvtx.RangePush("Pipeline: run: {}".format(self.run_count), self.run_count)
        num_stage = len(self.models)
        x.to_gpu()
        # Foward
        print("Pipeline: Forward")
        nvtx.RangePush("Pipeline: Forward", 1)
        next_input = x
        for i in range(0, num_stage):
            print("Pipeline: Forward: Stage:{} start".format(i))
            nvtx.RangePush("Pipeline: Forward: Stage: {}".format(i), i)

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
        print("Pipeline: Backward & Update")
        nvtx.RangePush("Pipeline: Backward & Update", 2)
        for i in reversed(range(0, num_stage)):

            print("Pipeline: Backward & Update: Stage:{} start".format(i))
            nvtx.RangePush("Pipeline: Backward & Update: Stage: {}".format(i), i)

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
        nvtx.RangePush("Pipeline: run: {}".format(self.run_count), self.run_count)
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
        nvtx.RangePush("Pipeline: Forward", 1)
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
            nvtx.RangePush("Pipeline: Forward: Stage: {}".format(i), i)
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
        nvtx.RangePush("Pipeline: Backward & Update", 2)
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
            nvtx.RangePush("Pipeline: Backward & Update: Stage: {}".format(i), i)
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

class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )
        self.train = True

    def __call__(self, x, t):
        h1 = F.max_pooling_2d(F.relu(F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h2 = F.max_pooling_2d(F.relu(F.local_response_normalization(self.conv2(h1))), 3, stride=2)
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.max_pooling_2d(F.relu(self.conv5(h4)), 3, stride=2)
        # h6 = F.dropout(F.relu(self.fc6(h5)), train=self.train)
        h6 = F.relu(self.fc6(h5))
        # h7 = F.dropout(F.relu(self.fc7(h6)), train=self.train)
        h7 = F.relu(self.fc7(h6))
        h8 = self.fc8(h7)
        loss = F.softmax_cross_entropy(h8, t)
        return loss

############################################################

class Alex_st1(chainer.Chain):
    def __init__(self):
        super(Alex_st1, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
        )
        self.train = True
    def __call__(self, x):
        # h1 = F.max_pooling_2d(F.relu(
        #     F.local_response_normalization(self.conv1(x))), 3, stride=2)
        # return h1
        h1_a = self.conv1(x)
        h1_a.name = "h1_a.conv1"
        h1_b = F.local_response_normalization(h1_a)
        h1_b.name = "h1_b.lrn"
        h1_c = F.relu(h1_b)
        h1_c.name = "h1_c.relu"
        h1 = F.max_pooling_2d(h1_c, 3, stride=2)
        h1.name = "h1.pooling"
        return h1

class Alex_st1_a(chainer.Chain):
    def __init__(self):
        super(Alex_st1_a, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
        )
        self.train = True
    def __call__(self, x):
        h1_a = self.conv1(x)
        h1_a.name = "h1_a.conv1"
        h1_b = F.local_response_normalization(h1_a)
        h1_b.name = "h1_b.lrn"
        return h1_b

class Alex_st1_b(chainer.Chain):
    def __init__(self):
        super(Alex_st1_b, self).__init__(
        )
        self.train = True
    def __call__(self, h1_b):
        h1_c = F.relu(h1_b)
        h1_c.name = "h1_c.relu"
        h1 = F.max_pooling_2d(h1_c, 3, stride=2)
        h1.name = "h1.pooling"
        return h1

class Alex_st2(chainer.Chain):
    def __init__(self):
        super(Alex_st2, self).__init__(
            conv2=L.Convolution2D(96, 256,  5, pad=2),
        )
        self.train = True
    def __call__(self, h1):
        # h2 = F.max_pooling_2d(F.relu(
        #     F.local_response_normalization(self.conv2(h1))), 3, stride=2)
        # return h2
        h2_a = self.conv2(h1)
        h2_a.name = "h2_a.conv2"
        h2_b = F.local_response_normalization(h2_a)
        h2_b.name = "h2_b.lrn"
        h2_c = F.relu(h2_b)
        h2_c.name = "h2_c.relu"
        h2 = F.max_pooling_2d(h2_c, 3, stride=2)
        h2.name = "h2.pooling"
        return h2

class Alex_st3(chainer.Chain):
    def __init__(self):
        super(Alex_st3, self).__init__(
            conv3=L.Convolution2D(256, 384,  3, pad=1),
        )
        self.train = True
    def __call__(self, h2):
        # h3 = F.relu(self.conv3(h2))
        h3_a = self.conv3(h2)
        h3_a.name = "h3_a.conv3"
        h3 = F.relu(h3_a)
        h3.name = "h3.relu"
        return h3

class Alex_st4(chainer.Chain):
    def __init__(self):
        super(Alex_st4, self).__init__(
            conv4=L.Convolution2D(384, 384,  3, pad=1),
        )
        self.train = True
    def __call__(self, h3):
        # h4 = F.relu(self.conv4(h3))
        h4_a = self.conv4(h3)
        h4_a.name = "h4_a.conv4"
        h4 = F.relu(h4_a)
        h4.name = "h4.relu"
        return h4

class Alex_st5(chainer.Chain):
    def __init__(self):
        super(Alex_st5, self).__init__(
            conv5=L.Convolution2D(384, 256,  3, pad=1),
        )
        self.train = True
    def __call__(self, h4):
        # h5 = F.max_pooling_2d(F.relu(self.conv5(h4)), 3, stride=2)
        h5_a = self.conv5(h4)
        h5_a.name = "h5_a.conv5"
        h5_b = F.relu(h5_a)
        h5_b.name = "h5_b.relu"
        h5 = F.max_pooling_2d(h5_b, 3, stride=2)
        h5.name = "h5.pooling"
        return h5

class Alex_st6(chainer.Chain):
    def __init__(self):
        super(Alex_st6, self).__init__(
            fc6=L.Linear(9216, 4096),
        )
        self.train = True
    def __call__(self, h5):
        # h6 = F.dropout(F.relu(self.fc6(h5)), train=self.train)
        h6_a = self.fc6(h5)
        h6_a.name = "h6_a.fc6"
        h6_b = F.relu(h6_a)
        h6_b.name = "h6_b.relu"
        # h6 = F.dropout(h6_b, train=self.train)
        # h6.name = "h6.dropout"
        # return h6
        return h6_b

class Alex_st7(chainer.Chain):
    def __init__(self):
        super(Alex_st7, self).__init__(
            fc7=L.Linear(4096, 4096),
        )
        self.train = True
    def __call__(self, h6):
        # h7 = F.dropout(F.relu(self.fc7(h6)), train=self.train)
        h7_a = self.fc7(h6)
        h7_a.name = "h7_a.fc7"
        h7_b = F.relu(h7_a)
        h7_b.name = "h7_b.relu"
        # h7 = F.dropout(h7_b, train=self.train)
        # h7.name = "h7.dropout"
        # return h7
        return h7_b

class Alex_st8(chainer.Chain):
    def __init__(self):
        super(Alex_st8, self).__init__(
            fc8=L.Linear(4096, 1000),
        )
        self.train = True
    def __call__(self, h7, t):
        h8 = self.fc8(h7)
        h8.name = "h8.fc8"
        loss = F.softmax_cross_entropy(h8, t)
        loss.name = "loss.sce"
        return loss

############################################################

model1 = Alex()
opt1 = optimizers.SGD()
opt1.setup(model1)

############################################################

#model_st1 = Alex_st1()
model_st1_a = Alex_st1_a()
model_st1_b = Alex_st1_b()
model_st2 = Alex_st2()
model_st3 = Alex_st3()
model_st4 = Alex_st4()
model_st5 = Alex_st5()
model_st6 = Alex_st6()
model_st7 = Alex_st7()
model_st8 = Alex_st8()
# 
pipeline = Pipeline(optimizers.SGD)
#pipeline.add_model( model_st1 )
pipeline.add_model( model_st1_a )
pipeline.add_model( model_st1_b )
pipeline.add_model( model_st2 )
pipeline.add_model( model_st3 )
pipeline.add_model( model_st4 )
pipeline.add_model( model_st5 )
pipeline.add_model( model_st6 )
pipeline.add_model( model_st7 )
pipeline.add_model( model_st8 )

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

# model_st1.conv1.W.copydata( model1.conv1.W )
# model_st1.conv1.b.copydata( model1.conv1.b )

model_st1_a.conv1.W.copydata( model1.conv1.W )
model_st1_a.conv1.b.copydata( model1.conv1.b )

model_st2.conv2.W.copydata( model1.conv2.W )
model_st2.conv2.b.copydata( model1.conv2.b )

model_st3.conv3.W.copydata( model1.conv3.W )
model_st3.conv3.b.copydata( model1.conv3.b )

model_st4.conv4.W.copydata( model1.conv4.W )
model_st4.conv4.b.copydata( model1.conv4.b )

model_st5.conv5.W.copydata( model1.conv5.W )
model_st5.conv5.b.copydata( model1.conv5.b )

model_st6.fc6.W.copydata( model1.fc6.W )
model_st6.fc6.b.copydata( model1.fc6.b )

model_st7.fc7.W.copydata( model1.fc7.W )
model_st7.fc7.b.copydata( model1.fc7.b )

model_st8.fc8.W.copydata( model1.fc8.W )
model_st8.fc8.b.copydata( model1.fc8.b )

# compare_links()

############################################################

# nbatch=64*1
# nbatch=64*2
# nbatch=64*3
# nbatch=64*4
# nbatch=64*5
# nbatch=64*6
# nbatch=64*7  # 1-stage: OK
nbatch=64*8
# nbatch=64*9
# nbatch=64*10
# nbatch=64*11
# nbatch=64*12
# nbatch=64*13  # 8-stage: OK
# nbatch=64*14
# nbatch=64*15  # 9-stage: OK
# nbatch=64*16


x = xp.random.uniform(-1, 1, (nbatch, 3, 227, 227)).astype(xp.float32)
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

num_loop = 10

############################################################

if False:
    print '#################### unified model ####################'
    if (use_GPU):
        model1.to_gpu()
    
    for loop in range(0, num_loop):
    
        runtime.deviceSynchronize()
        nvtx.RangePush("Unified: {}".format(loop), loop)
    
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
    print '#################### split model ####################'
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

