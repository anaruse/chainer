from chainer import function

import sys

class FusedFunction(function.Function):

    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def forward(self, inputs):
        # inputs: tuple of ndarray
        print("[fused_function.py, forward()]")
        print("    inputs: type:{}, len:{}".format(type(inputs),len(inputs)))

        f1_inputs = tuple([inputs[0]])

        hs = self.f1.forward(f1_inputs)

        f2_inputs = [hs[0]]
        for i in inputs[1:]:
            f2_inputs.append(i)
        f2_inputs = tuple(f2_inputs)

        outputs = self.f2.forward(f2_inputs)
        print("    outputs: type:{}, len:{}".format(type(outputs),len(outputs)))

        # work-around for ReLU
        if hasattr(self.f1, 'y'):
            self.f1.y = None
        return outputs

    def backward(self, inputs, g_outputs):
        # inputs: tuple of ndarray
        # g_outputs: tuple of ndarray
        print("[fused_function.py, backward()]")
        print("    inputs: type:{}, len:{}".format(type(inputs),len(inputs)))
        print("    g_outputs: type:{}, len:{}".format(type(g_outputs),len(g_outputs)))

        f1_inputs = tuple([inputs[0]])
        # print("    f1_inputs: type:{}, len:{}".format(type(f1_inputs),len(f1_inputs)))

        f1_outputs = self.f1.forward(f1_inputs)
        # print("    f1_outputs: type:{}, len:{}".format(type(f1_outputs),len(f1_outputs)))

        f2_inputs = [f1_outputs[0]]
        for i in inputs[1:]:
            f2_inputs.append(i)
        f2_inputs = tuple(f2_inputs)
        # print("    f2_inputs: type:{}, len:{}".format(type(f2_inputs),len(f2_inputs)))

        g_f2_outputs = g_outputs
        g_f2_inputs = self.f2.backward(f2_inputs, g_f2_outputs)
        # print("    g_f2_inputs: type:{}, len:{}".format(type(g_f2_inputs),len(g_f2_inputs)))

        g_f1_outputs = tuple([g_f2_inputs[0]])
        # print("    g_f1_outputs: type:{}, len:{}".format(type(g_f1_outputs),len(g_f1_outputs)))

        g_f1_inputs = self.f1.backward(f1_inputs, g_f1_outputs)
        # print("    g_f1_inputs: type:{}, len:{}".format(type(g_f1_inputs),len(g_f1_inputs)))

        g_inputs = [g_f1_inputs[0]]
        for i in g_f2_inputs[1:]:
            g_inputs.append(i)
        g_inputs = tuple(g_inputs)
        print("    g_inputs: type:{}, len:{}".format(type(g_inputs),len(g_inputs)))

        self.f1 = None
        self.f2 = None

        return g_inputs
