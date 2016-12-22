from chainer import function

import sys

class FusedFunction(function.Function):

    def __init__(self, *funcs):
        print("[fused_function.py, __init__()]")
        print("    funcs: type:{}, len:{}".format(type(funcs),len(funcs)))

        n_funcs = len(funcs)

        self.pre_funcs = funcs[:n_funcs-1]  # tuple of function object
        self.core_func = funcs[n_funcs-1]   # function object

        print("    pre_funcs: type:{}, len:{}".format(type(self.pre_funcs),len(self.pre_funcs)))

    def forward(self, inputs):
        # inputs: tuple of ndarray
        print("[fused_function.py, forward()]")
        print("    inputs: type:{}, len:{}".format(type(inputs),len(inputs)))
        # print("    inputs[0]: type:{}, size:{}".format(type(inputs[0]),inputs[0].size))

        n_pre_funcs = len(self.pre_funcs)

        hs = tuple([inputs[0]])
        # print("    hs: type:{}, len:{}".format(type(hs),len(hs)))
        # print("    hs[0]: type:{}, size:{}".format(type(hs[0]),hs[0].size))
        for i in range(n_pre_funcs):
            pre_func = self.pre_funcs[i]
            print("    pre_funcs[{}]:{}".format(i,pre_func))
            hs = pre_func.forward(hs)
            # print("    hs: type:{}, len:{}".format(type(hs),len(hs)))
            # print("    hs[0]: type:{}, size:{}".format(type(hs[0]),hs[0].size))

            # work-around for ReLU
            if hasattr(pre_func, 'y'):
                pre_func.y = None
            
        core_inputs = [hs[0]]
        for i in inputs[1:]:
            core_inputs.append(i)
        core_inputs = tuple(core_inputs)
        # print("    core_inputs: type:{}, len:{}".format(type(core_inputs),len(core_inputs)))

        print("    core_func:{}".format(self.core_func))
        outputs = self.core_func.forward(core_inputs)
        print("    outputs: type:{}, len:{}".format(type(outputs),len(outputs)))
        # print("    outputs[0]: type:{}, size:{}".format(type(outputs[0]),outputs[0].size))

        return outputs

    def backward(self, inputs, g_outputs):
        # inputs: tuple of ndarray
        # g_outputs: tuple of ndarray
        print("[fused_function.py, backward()]")
        print("    inputs: type:{}, len:{}".format(type(inputs),len(inputs)))
        print("    g_outputs: type:{}, len:{}".format(type(g_outputs),len(g_outputs)))


        n_pre_funcs = len(self.pre_funcs)
        hs_list = []

        hs = tuple([inputs[0]])
        hs_list.append(hs)
        for i in range(n_pre_funcs):
            pre_func = self.pre_funcs[i]
            print("    pre_funcs[{}]:{}".format(i,pre_func))
            hs = pre_func.forward(hs)
            hs_list.append(hs)
            # print("    hs: type:{}, len:{}".format(type(hs),len(hs)))
            # print("    hs[0]: type:{}, size:{}".format(type(hs[0]),hs[0].size))

        hs = hs_list.pop()
        core_inputs = [hs[0]]
        for i in inputs[1:]:
            core_inputs.append(i)
        core_inputs = tuple(core_inputs)
        # print("    core_inputs: type:{}, len:{}".format(type(core_inputs),len(core_inputs)))

        print("    core_func:{}".format(self.core_func))
        g_core_inputs = self.core_func.backward(core_inputs, g_outputs)

        g_hs = tuple([g_core_inputs[0]])
        for i in reversed(range(n_pre_funcs)):
            pre_func = self.pre_funcs[i]
            print("    pre_funcs[{}]:{}".format(i,pre_func))
            hs = hs_list.pop()
            g_hs = pre_func.backward(hs, g_hs)

        g_inputs = [g_hs[0]]
        for i in g_core_inputs[1:]:
            g_inputs.append(i)
        g_inputs = tuple(g_inputs)
        print("    g_inputs: type:{}, len:{}".format(type(g_inputs),len(g_inputs)))

        self.pre_funcs = None
        self.core_func = None

        return g_inputs
