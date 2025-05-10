import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################################################################################################

class CSVReader:
    def __init__(self, filePath: str) -> None:
        self.filePath = filePath
        self.data = None
        self.labels = None
        self.featureNames = None
        self.labelName = None
        self.__loadData__()

    def __loadData__(self) -> None:
        """加载数据并初始化属性"""
        data, labels = [], []
        with open(self.filePath) as file:
            reader = csv.reader(file)
            headers = next(reader)
            self.featureNames = headers[:-1]
            self.labelName = headers[-1]
            
            for row in reader:
                processedRow = [float(x) if x else np.nan for x in row[:-1]]
                data.append(processedRow)
                labels.append(row[-1])
        
        le = LabelEncoder()
        self.data = np.array(data)
        self.labels = le.fit_transform(labels)
    
    def getData(self) -> tuple[np.ndarray, np.ndarray]:
        """获取特征数据和标签"""
        return self.data, self.labels
    
    def checkBasicInfo(self) -> None:
        """检查数据基本信息"""
        print("\n=== 数据基本信息 ===")
        print(f"特征数: {len(self.featureNames)}")
        print(f"样本数: {len(self.data)}")
        print(f"特征名称: {self.featureNames}")
        print(f"标签名称: {self.labelName}")
        
        nanCount = np.sum(np.isnan(self.data))
        print(f"\n缺失值总数: {nanCount}")
        if nanCount > 0:
            print("各特征缺失值数量:")
            for i, name in enumerate(self.featureNames):
                print(f"{name}: {np.sum(np.isnan(self.data[:, i]))}")
    
    def showStatistics(self) -> None:
        """显示数据统计信息"""
        if self.data is None:
            raise ValueError("数据未加载！")
        
        print("\n=== 数据统计信息 ===")
        stats = {
            'mean': np.nanmean(self.data, axis=0),
            'std': np.nanstd(self.data, axis=0),
            'min': np.nanmin(self.data, axis=0),
            '25%': np.nanpercentile(self.data, 25, axis=0),
            'median': np.nanmedian(self.data, axis=0),
            '75%': np.nanpercentile(self.data, 75, axis=0),
            'max': np.nanmax(self.data, axis=0)
        }
        
        statsDf = pd.DataFrame(stats, index=self.featureNames)
        print(statsDf.round(2))
    
    @staticmethod
    def read(filePath: str) -> tuple[np.ndarray, np.ndarray]:
        """静态快速加载方法"""
        instance = CSVReader(filePath)
        return instance.getData()
    
#####################################################################################################

class Math:
    def normalize(data: np.ndarray) -> np.ndarray:
        """
        按列归一化数据（每个特征独立归一化到[0,1]）
        参数:
            data: 二维数组，形状为 (n_samples, n_features)，每列代表一个特征
        返回:
            归一化后的数组，形状与输入相同
        """
        if data is None or data.size == 0:
            raise ValueError("输入数据不能为空！")
        
        if data.ndim != 2:
            raise ValueError("输入数据必须是二维数组（n_samples, n_features）")
        
        minVal = np.nanmin(data, axis=0, keepdims=True)
        maxVal = np.nanmax(data, axis=0, keepdims=True)
        rangeVal = maxVal - minVal
        rangeVal[rangeVal == 0] = 1.0  # 避免除以零
        
        normalizedData = (data - minVal) / rangeVal
        normalizedData[np.isnan(data)] = np.nan
        return normalizedData
    
    @staticmethod
    def divide(data: np.ndarray, labels: np.ndarray, ratio: float) -> tuple:
        """
        随机划分训练集和验证集
        参数:
            data: 输入特征数据 (n_samples, n_features)
            labels: 对应标签 (n_samples,)
            ratio: 验证集占比 (0 < ratio < 1)
        返回:
            (训练数据, 训练标签, 验证数据, 验证标签)
        """
        if ratio <= 0 or ratio >= 1:
            raise ValueError("ratio必须在0和1之间")
        
        # 验证输入形状
        assert data.ndim == 2, f"数据必须是二维数组，当前维度: {data.ndim}"
        assert labels.ndim == 1, f"标签必须是一维数组，当前维度: {labels.ndim}"
        assert len(data) == len(labels), "数据与标签数量不匹配"

        import pyvqnet.tensor as tensor
        
        # 转换为numpy处理，避免Qtensor的一些问题
        data_np = data.numpy()
        labels_np = labels.numpy()
        
        # 随机划分
        indices = np.random.permutation(len(data_np))
        split = int(len(indices) * ratio)
        
        # 划分数据
        train_data = tensor.QTensor(data_np[indices[split:]], dtype=data.dtype)
        train_labels = tensor.QTensor(labels_np[indices[split:]], dtype=labels.dtype)
        val_data = tensor.QTensor(data_np[indices[:split]], dtype=data.dtype)
        val_labels = tensor.QTensor(labels_np[indices[:split]], dtype=labels.dtype)
        
        return train_data, train_labels, val_data, val_labels

#####################################################################################################

class Draw:
    @staticmethod
    def curve(x: np.ndarray, y: np.ndarray, title: str, xLabel: str, yLabel: str) -> None:
        """
        绘制曲线图，适用于单个数据集。
        参数:
            x: x轴数据
            y: y轴数据
            title: 图表标题
        """
        
        plt.plot(x, y, label=title)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid()
        plt.show()
    
    @staticmethod
    def grid(shape: tuple[int, int], x: list[np.ndarray], y: list[np.ndarray], title: list[str], xLabel: list[str], yLabel: list[str]) -> None:
        """
        绘制网格图，适用于多张子图的布局。
        参数:
            shape: 网格形状 (rows, cols)
            x: x轴数据列表，每个元素是一个一维数组
            y: y轴数据列表，每个元素是一个一维数组
            title: 每个子图的标题列表
            xLabel: 每个子图的x轴标签列表
            yLabel: 每个子图的y轴标签列表
        """
        for i in range(min(shape[0] * shape[1], len(x), len(y), len(title))):
            plt.subplot(shape[0], shape[1], i + 1)
            plt.plot(x[i], y[i])
            plt.title(title[i])
            plt.xlabel(xLabel[i])
            plt.ylabel(yLabel[i])
            plt.grid()
        plt.show()

#####################################################################################################

if __name__ == "__main__":
    filePath = "Data/train_data.csv"
    trainReader = CSVReader(filePath)
    print(trainReader.featureNames)
    trainReader.checkBasicInfo()
    trainReader.showStatistics()
