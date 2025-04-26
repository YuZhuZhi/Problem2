import Utils
import numpy as np
import pyvqnet
from pyvqnet.tensor import *

#####################################################################################################

#####################################################################################################

if __name__ == "__main__":
    trainData, trainLabels = Utils.CSVReader.read("train_data.csv")
    testData, testLabels = Utils.CSVReader.read("test_data.csv")
    print(trainLabels)
    print(testLabels)
