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

class SmallResNet(vqn.nn.Module):
    class BasicBlock(vqn.nn.Module):
        def __init__(self, inChannels: int, hiddenDim: int) -> None:
            super(SmallResNet.BasicBlock, self).__init__()
            self.linear1 = Linear(inChannels, hiddenDim, dtype=kfloat32)
            self.linear2 = Linear(hiddenDim, inChannels, dtype=kfloat32)
            self.relu = ReLu()
        
        def forward(self, x: QTensor) -> QTensor:
            # x.to(DEVICE)
            residual = x
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            x = x + residual
            return self.relu(x)
            
    #------------------------------------------------------------------------------#
    
    def __init__(self, inChannels: int=9, outChannels: int=4) -> None:
        super(SmallResNet, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        
        self.inputLayer = Linear(inChannels, 8, dtype=kfloat32)
        self.residualBlock = SmallResNet.BasicBlock(8, 4)
        self.outputLayer = Linear(8, outChannels, dtype=kfloat32)
        self.relu = ReLu()
        self.softmax = Softmax()
        
    def forward(self, x: QTensor) -> QTensor:
        # x.to(DEVICE)
        x = self.relu(self.inputLayer(x))
        x = self.residualBlock(x)
        x = self.outputLayer(x)
        return self.softmax(x)
    
#####################################################################################################

class SmallMLP(vqn.nn.Module):
    def __init__(self, inChannels: int = 9, outChannels: int = 4) -> None:
        super(SmallMLP, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        
        self.layer1 = Linear(inChannels, 8, dtype=kfloat32)
        self.layer2 = Linear(8, 6, dtype=kfloat32)
        self.outputLayer = Linear(6, outChannels, dtype=kfloat32)
        self.relu = ReLu()
        self.softmax = Softmax()
        
    def forward(self, x: QTensor) -> QTensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.outputLayer(x)
        return self.softmax(x)
    
#####################################################################################################

if __name__ == "__main__":
    model = SmallResNet()
    # model.to_gpu()
    print(model.parameters())
    # print(f"总参数量: {sum(p.numel() for p in model.parameters())}")

