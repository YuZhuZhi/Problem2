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
    def __init__(self, model: pyvqnet.nn.Module, data: np.ndarray, labels: np.ndarray, val_ratio: float=0.2) -> None:
        self.model = model
        self.data = QTensor(Utils.Math.normalize(data), dtype=kfloat32)
        self.labels = QTensor(labels, dtype=pyvqnet.kint64)
        self.lossFunction = pyvqnet.nn.CrossEntropyLoss()
        # self.lossFunction = pyvqnet.nn.NLL_Loss()
        self.optimizer = pyvqnet.optim.Adam(self.model.parameters(), lr=INITIAL_LEARNING_RATE)
        # set_random_seed(SEED)
        self.trainLoss, self.trainAcc = [], []
        self.valLoss, self.valAcc, self.valF1 = [], [], []
        self.val_ratio = val_ratio

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
    
    def train2(self, epochs: int) -> None:
        """重构后的训练函数"""
        print(f"\n开始训练 | 总轮次: {epochs} | 验证比例: {self.val_ratio*100}%")
        
        for epoch in range(epochs):
            # 执行单步训练并获取指标
            train_loss, train_acc, val_loss, val_acc, val_f1 = self.oneStepTrain2()
            
            # 记录所有指标
            self.trainLoss.append(train_loss)
            self.trainAcc.append(train_acc)
            self.valLoss.append(val_loss)
            self.valAcc.append(val_acc)
            self.valF1.append(val_f1)
            
            # 进度打印
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f} | "
                  f"Val F1: {val_f1:.4f}")
            
    def oneStepTrain2(self) -> tuple:
        """整合数据划分的单步训练验证流程"""
        # 动态划分数据集
        train_data, train_labels, val_data, val_labels = Utils.Math.divide(
            self.data, 
            self.labels,
            self.val_ratio
        )

        # --- 训练阶段 ---
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(train_data)
        loss = self.lossFunction(train_labels, outputs)
        loss.backward()
        self.optimizer._step()
        
        # 训练指标计算
        train_pred = outputs.argmax(dim=1)
        train_acc = (train_pred == train_labels).float().mean().item()
        
        # --- 验证阶段 ---
        self.model.eval()
        with no_grad():
            val_outputs = self.model(val_data)
            val_loss = self.lossFunction(val_labels, val_outputs).item()
            
            val_pred = val_outputs.argmax(dim=1)
            val_acc = (val_pred == val_labels).float().mean().item()
            _, _, val_f1 = metrics.precision_recall_f1_N_score(
                val_labels, val_pred, 4, average="macro"
            )
        
        return (loss.item(), train_acc, val_loss, val_acc, val_f1)

            
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
        
    def visualize(self):
        """可视化训练过程"""
        Utils.Draw.grid(
            shape=[2, 2],
            x=[np.arange(len(self.trainLoss))]*5,
            y=[self.trainLoss, self.trainAcc, self.valLoss, self.valAcc, self.valF1],
            title=["Training Loss", "Training Accuracy", 
                   "Validation Loss", "Validation Accuracy", "Validation F1 Score"],
            xLabel=["Step"]*5,
            yLabel=["Loss", "Accuracy", "Loss", "Accuracy", "F1 Score"]
        )

#####################################################################################################

if __name__ == "__main__":
    pass