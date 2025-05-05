import Utils
import numpy as np
import pyvqnet
from pyvqnet.tensor import *
from pyvqnet.utils import metrics, get_random_seed, set_random_seed

#####################################################################################################
INITIAL_LEARNING_RATE = 1e-2
SEED = 51346
#####################################################################################################

class Trainer:
    def __init__(self, model: pyvqnet.nn.Module, data: np.ndarray, labels: np.ndarray) -> None:
        self.model = model
        self.data = QTensor(Utils.Math.normalize(data), dtype=kfloat32)
        self.labels = QTensor(labels, dtype=pyvqnet.kint64)
        self.lossFunction = pyvqnet.nn.CrossEntropyLoss()
        # self.lossFunction = pyvqnet.nn.NLL_Loss()
        self.optimizer = pyvqnet.optim.Adam(self.model.parameters(), lr=INITIAL_LEARNING_RATE)
        # set_random_seed(SEED)
        self.trainLoss, self.trainAcc = [], []

    def train(self, epochs: int) -> None:
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.data)
            loss = self.lossFunction(self.labels, outputs)
            loss.backward()
            self.optimizer._step()
            prediction = outputs.argmax(dim=1)
            accuracy = (prediction == self.labels).float().mean().item()
            self.trainLoss.append(loss.item())
            self.trainAcc.append(accuracy)
            precision, recall, f1score = metrics.precision_recall_f1_N_score(self.labels, prediction, 4, average="macro")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.4f}, F1 Score: {f1score:.4f}")

    def oneStepTrain(self) -> tuple:
        """
        执行单步训练并返回指标
        返回: 
            (loss, accuracy, precision, recall, f1score)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向计算
        outputs = self.model(self.data)
        loss = self.lossFunction(self.labels, outputs)
        
        # 反向传播
        loss.backward()
        self.optimizer._step()
        
        # 计算指标
        prediction = outputs.argmax(dim=1)
        accuracy = (prediction == self.labels).float().mean().item()
        precision, recall, f1 = metrics.precision_recall_f1_N_score(
            self.labels, prediction, 4, average="macro")
        
        # 记录训练过程
        self.trainLoss.append(loss.item())
        self.trainAcc.append(accuracy)
        
        return (loss.item(), accuracy, precision, recall, f1)
            
    def test(self, testData: np.ndarray, testLabels: np.ndarray, outputFile: str=None) -> np.ndarray:
        self.model.eval()
        testDataTensor = QTensor(Utils.Math.normalize(testData), dtype=kfloat32)
        testLabelsTensor = QTensor(testLabels, dtype=pyvqnet.kint64)
        with no_grad():
            outputs = self.model(testDataTensor)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == testLabelsTensor).float().mean().item()
            precision, recall, f1score = metrics.precision_recall_f1_N_score(testLabelsTensor, predictions, 4, average="macro")
            print(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1score:.4f}")
            print(get_random_seed())
            if outputFile is not None:
                with open(outputFile, "a") as file:
                    file.write(f"{accuracy:.4f} {f1score:.4f}\n")
            return predictions.numpy()
#####################################################################################################

if __name__ == "__main__":
    pass