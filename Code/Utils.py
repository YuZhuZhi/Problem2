import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################################################################################################

class CSVReader:
    @staticmethod
    def read(fileName: str) -> tuple[np.ndarray, np.ndarray]:
        data, labels = [], []
        with open(fileName) as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                data.append(list(map(float, row[:-1])))
                labels.append(row[-1])
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        return np.array(data), np.array(encoded_labels)