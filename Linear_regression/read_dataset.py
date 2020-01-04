import ntpath
import re
import numpy as np

class DataSet:
    def __init__(self, file_name):
        file = open(file_name)
        head, self.data_set_name = ntpath.split(file_name)
        self.data_set_name = re.sub('\.(.*)', '', self.data_set_name)
        print(self.data_set_name)

        self.feature_name = file.readline().split(',')
        self.target_name = self.feature_name[-1]
        self.feature_name = self.feature_name[:-1]

        self.samples_values = file.readlines()
        self.target_feature = []

        for i in range(len(self.samples_values)):
            self.samples_values[i] = re.sub('\+,', '', self.samples_values[i])
            self.samples_values[i] = self.samples_values[i].strip().split(',')
            for j in range(len(self.samples_values[i])):
                if self.samples_values[i][j].lstrip('-').replace('.', '', 1).isdigit():
                    self.samples_values[i][j] = float(self.samples_values[i][j])
            self.target_feature.append(self.samples_values[i].pop())

    def getDataSetAsNumPy(self):
        samples_values_arr = np.array(self.samples_values, ndmin=2)
        target_feature_arr = np.array(self.target_feature, ndmin=2)

        print(samples_values_arr)
        print(target_feature_arr)

        return samples_values_arr, target_feature_arr




