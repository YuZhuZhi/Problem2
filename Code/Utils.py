import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

#####################################################################################################

if __name__ == "__main__":
    filePath = "Data/train_data.csv"
    trainReader = CSVReader(filePath)
    print(trainReader.featureNames)
    trainReader.checkBasicInfo()
    trainReader.showStatistics()
