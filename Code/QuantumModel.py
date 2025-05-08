import Utils
import numpy as np

import pyvqnet as vqn
from pyvqnet.tensor import *
from pyvqnet.tensor import arange
from pyvqnet import kfloat32, DEV_GPU_0
from pyvqnet.nn import Linear, ReLu, Softmax
from pyvqnet.qnn.vqc import QMachine, QModule, rot, crx, rx, MeasureAll

#####################################################################################################
DEVICE = DEV_GPU_0
#####################################################################################################

class buildQMLP(QModule):
    def __init__(self, numQubits: int) -> None:
        super(buildQMLP, self).__init__()
        self.numQubits = numQubits
        self.quantumMachine = QMachine(numQubits)
        self.weights = vqn.nn.Parameter((numQubits * 8, ))
        pauliStrList = []
        for position in range(numQubits):
            pauliStrList.append({"Z" + str(position): 1.0})            
        self.measureAll = MeasureAll(obs=pauliStrList)
        
    def rotCircuit(self, weights) -> None:
        for i in range(self.numQubits):
            rot(q_machine=self.quantumMachine,
                wires=i,
                params=weights[3 * i: 3 * i + 3])
    
    def controlRotCircuit(self, weights) -> None:
        for i in range(self.numQubits):
            crx(q_machine=self.quantumMachine,
                wires=[i, (i + 1) % self.numQubits],
                params=weights[i])
        
    def forward(self, x: QTensor) -> QTensor:
        self.quantumMachine.reset_states(x.shape[0])
        for i in range(self.numQubits):
            rx(self.quantumMachine, i, x[:, [i]])
        self.rotCircuit(self.weights[0: 3 * self.numQubits])
        self.controlRotCircuit(self.weights[3 * self.numQubits: 4 * self.numQubits])
        for i in range(self.numQubits):
            rx(self.quantumMachine, i, x[:, [i]])
        self.rotCircuit(self.weights[4 * self.numQubits: 7 * self.numQubits])
        self.controlRotCircuit(self.weights[7 * self.numQubits: 8 * self.numQubits])
        return self.measureAll(self.quantumMachine)

#####################################################################################################

class QuantumMLP(vqn.nn.Module):
    def __init__(self, inChannels: int = 9, outChannels: int = 4, numQubits: int = 4) -> None:
        super(QuantumMLP, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.numQubits = numQubits
        
        self.inputLayer = Linear(inChannels, numQubits, dtype=kfloat32)
        self.quantumModel = buildQMLP(numQubits)
        
    def forward(self, x: QTensor) -> QTensor:
        x = self.inputLayer(x)
        quantumResult = self.quantumModel(x)
        return quantumResult
