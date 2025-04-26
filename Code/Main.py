import Utils
import ClassicModel
import Trainer
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################################################################################################
TRAIN_SET_PATH = "Data/train_data.csv"
TEST_SET_PATH = "Data/test_data.csv"
#####################################################################################################

if __name__ == "__main__":
    trainReader = Utils.CSVReader(TRAIN_SET_PATH)
    testReader = Utils.CSVReader(TEST_SET_PATH)
    
    smallResNet = ClassicModel.SmallResNet(inChannels=9, outChannels=4)
    smallMLP = ClassicModel.SmallMLP(inChannels=9, outChannels=4)
    
    classicTrainer = Trainer.Trainer(model=smallMLP, data=trainReader.data, labels=trainReader.labels)
    classicTrainer.train(epochs=100)
    classicTrainer.test(testReader.data, testReader.labels)
    