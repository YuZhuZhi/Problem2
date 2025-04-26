import Utils
import numpy as np
import pyvqnet
from pyvqnet.tensor import *
from pyvqnet.utils import metrics

#####################################################################################################
INITIAL_LEARNING_RATE = 1e-2
#####################################################################################################

class Trainer:
    def __init__(self, model: pyvqnet.nn.Module, data: np.ndarray, labels: np.ndarray) -> None:
        self.model = model
        self.data = QTensor(data, dtype=kfloat32)
        self.labels = QTensor(labels, dtype=pyvqnet.kint64)
        self.lossFunction = pyvqnet.nn.CrossEntropyLoss()
        self.optimizer = pyvqnet.optim.Adam(self.model.parameters(), lr=INITIAL_LEARNING_RATE)
    
    def train(self, epochs: int) -> None:
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.data)
            loss = self.lossFunction(self.labels, outputs)
            loss.backward()
            self.optimizer._step()
            prediction = outputs.argmax(dim=1)
            precision, recall, f1score = metrics.precision_recall_f1_N_score(self.labels, prediction, 4, average="macro")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Precision: {precision:.4f}, F1 Score: {f1score:.4f}")
            
    def test(self, testData: np.ndarray, testLabels: np.ndarray) -> np.ndarray:
        self.model.eval()
        testDataTensor = QTensor(testData, dtype=kfloat32)
        testLabelsTensor = QTensor(testLabels, dtype=pyvqnet.kint64)
        with no_grad():
            outputs = self.model(testDataTensor)
            predictions = outputs.argmax(dim=1)
            
            precision, recall, f1score = metrics.precision_recall_f1_N_score(testLabelsTensor, predictions, 4, average="macro")
            print(f"Test Precision: {precision}, F1 Score: {f1score}")
            
            return predictions.numpy()
#####################################################################################################

if __name__ == "__main__":
    pass