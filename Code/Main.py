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
EPOCHS = 250
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
    
    # Utils.Draw.grid(shape=[1, 2], x=[np.arange(0, EPOCHS), np.arange(0, EPOCHS)], 
    #                 y=[classicTrainer.trainAcc, classicTrainer.trainLoss], 
    #                 title=["Train Accuracy", "Train Loss"], 
    #                 xLabel=["Epoch", "Epoch"], 
    #                 yLabel=["Accuracy", "Loss"])
    
    # # 可供参考的新函数使用方式
    # classicTrainer = Trainer.Trainer(model=smallMLP, data=trainReader.data, labels=trainReader.labels, val_ratio=VAL_RATIO)
    # classicTrainer.train2(epochs=EPOCHS)
    # classicTrainer.test(testReader.data, testReader.labels, outputFile=OUTPUT_PATH)   
    # classicTrainer.visualize()
    
    qmlp = QuantumModel.QuantumMLP(inChannels=9, outChannels=4, numQubits=4)
    quantumTrainer = Trainer.Trainer(model=qmlp, data=trainReader.data, labels=trainReader.labels)
    quantumTrainer.train(epochs=EPOCHS)
    quantumTrainer.test(testReader.data, testReader.labels, outputFile=OUTPUT_PATH)
    
    # Utils.Draw.grid(shape=[1, 2], x=[np.arange(0, EPOCHS), np.arange(0, EPOCHS)], 
    #                 y=[quantumTrainer.trainAcc, quantumTrainer.trainLoss], 
    #                 title=["Train Accuracy", "Train Loss"], 
    #                 xLabel=["Epoch", "Epoch"], 
    #                 yLabel=["Accuracy", "Loss"])
    
    # # 合并显示Loss和Accuracy对比图
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(np.arange(EPOCHS), classicTrainer.trainLoss, label='Classic')
    # plt.plot(np.arange(EPOCHS), quantumTrainer.trainLoss, label='Quantum')
    # plt.title("Training Loss Comparison")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(np.arange(EPOCHS), classicTrainer.trainAcc, label='Classic')
    # plt.plot(np.arange(EPOCHS), quantumTrainer.trainAcc, label='Quantum')
    # plt.title("Training Accuracy Comparison")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()  # 自动调整子图间距
    # plt.show()