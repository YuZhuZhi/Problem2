import Utils
import numpy as np

import pyvqnet as vqn
from pyvqnet.tensor import *
from pyvqnet.tensor import arange
from pyvqnet import kfloat32, DEV_GPU_0
from pyvqnet.nn import Linear, ReLu, Softmax

#####################################################################################################
DEVICE = DEV_GPU_0
#####################################################################################################

class QuantumMLP(vqn.nn.vqc.QModule):
    def __init__(self, inChannels: int = 9, outChannels: int = 4) -> None:
        super(QuantumMLP, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        
        self.layer1 = Linear(inChannels, 8, dtype=kfloat32)
        self.layer2 = Linear(8, 6, dtype=kfloat32)
        self.outputLayer = Linear(6, outChannels, dtype=kfloat32)
        self.relu = ReLu()
        self.softmax = Softmax()
