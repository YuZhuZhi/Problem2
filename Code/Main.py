import Utils
import ClassicModel
import QuantumModel
import Trainer
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################################################################################################
TRAIN_SET_PATH = "Data/train_data.csv"
TEST_SET_PATH = "Data/test_data.csv"
OUTPUT_PATH = "output.txt"
EPOCHS = 50
VAL_RATIO = 0.2
#####################################################################################################

if __name__ == "__main__":
    trainReader = Utils.CSVReader(TRAIN_SET_PATH)
    testReader = Utils.CSVReader(TEST_SET_PATH)
    
    smallResNet = ClassicModel.SmallResNet(inChannels=9, outChannels=4)
    # smallMLP = ClassicModel.SmallMLP(inChannels=9, outChannels=4)
    
    classicTrainer = Trainer.Trainer(model=smallResNet, data=trainReader.data, labels=trainReader.labels)
    classicTrainer.train(epochs=EPOCHS)
    classicTrainer.test(testReader.data, testReader.labels, outputFile=OUTPUT_PATH)
    
    Utils.Draw.grid(shape=[1, 2], x=[np.arange(0, EPOCHS), np.arange(0, EPOCHS)], 
                    y=[classicTrainer.trainAcc, classicTrainer.trainLoss], 
                    title=["Train Accuracy", "Train Loss"], 
                    xLabel=["Epoch", "Epoch"], 
                    yLabel=["Accuracy", "Loss"])
    
    # # 可供参考的新函数使用方式
    # classicTrainer = Trainer.Trainer(model=smallMLP, data=trainReader.data, labels=trainReader.labels, val_ratio=VAL_RATIO)
    # classicTrainer.train2(epochs=EPOCHS)
    # classicTrainer.test(testReader.data, testReader.labels, outputFile=OUTPUT_PATH)   
    # classicTrainer.visualize()
    
    # qmlp = QuantumModel.QuantumMLP(inChannels=9, outChannels=4, numQubits=4)
    # quantumTrainer = Trainer.Trainer(model=qmlp, data=trainReader.data, labels=trainReader.labels)
    # quantumTrainer.train(epochs=500)
    # quantumTrainer.test(testReader.data, testReader.labels, outputFile=OUTPUT_PATH)
    
    # Utils.Draw.grid(shape=[1, 2], x=[np.arange(0, EPOCHS), np.arange(0, EPOCHS)], 
    #                 y=[quantumTrainer.trainAcc, quantumTrainer.trainLoss], 
    #                 title=["Train Accuracy", "Train Loss"], 
    #                 xLabel=["Epoch", "Epoch"], 
    #                 yLabel=["Accuracy", "Loss"])
    