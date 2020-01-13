import read_dataset
from Regression_Model import LogisticRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def Test1():
    dataSet = read_dataset.DataSet('./DataSet/Generator_status.csv')
    learningRate = 0.5
    numofIteration = 500

    samples_values_arr, target_feature_arr = dataSet.getDataSetAsNumPy()
    target_feature_arr = pd.DataFrame(target_feature_arr)
    target_val_cat = target_feature_arr.iloc[:,0].unique().tolist()
    target_feature_arr = target_feature_arr.replace(target_val_cat, range(len(target_val_cat)))
    # print(target_feature_arr)

    logReg = LogisticRegression(samples_values_arr, target_feature_arr)
    model_weights, regressionLine = logReg.gradientDescent(initlearningRate=learningRate,
                                                            numofIteration=numofIteration)

    plt.figure(1)


    plt.scatter(samples_values_arr.iloc[:,0], samples_values_arr.iloc[:,1], s=10 * target_feature_arr,
                label='faulty', marker='x')
    plt.scatter(samples_values_arr.iloc[:,0], samples_values_arr.iloc[:,1], s=10 * (1 - target_feature_arr),
                label='good', marker='o')
    plt.xlabel(dataSet.dataSet.columns.values[0])
    plt.ylabel(dataSet.dataSet.columns.values[1])

    print(regressionLine)
    plt.plot(samples_values_arr.iloc[:,0], regressionLine, color='r')
    plt.title(dataSet.data_set_name + ' data set')
    # print(logReg.square_error)
    #
    plt.figure(2)
    plt.plot(range(numofIteration), logReg.square_error)
    plt.xlabel('Num of iterations')
    plt.ylabel('Square Error')
    plt.title('Cost Function over iteration')

    plt.legend()
    plt.show()



def Test2():
    dataSet = read_dataset.DataSet('./DataSet/Exams_grad.csv')
    learningRate = 0.5
    numofIteration = 500

    samples_values_arr, target_feature_arr = dataSet.getDataSetAsNumPy()
    target_feature_arr = pd.DataFrame(target_feature_arr)

    logReg = LogisticRegression(samples_values_arr, target_feature_arr)
    model_weights, regressionLine = logReg.gradientDescent(initlearningRate=learningRate,
                                                            numofIteration=numofIteration)

    plt.figure(1)


    plt.scatter(samples_values_arr.iloc[:,0], samples_values_arr.iloc[:,1], s=10 * target_feature_arr,
                label='faulty', marker='x')
    plt.scatter(samples_values_arr.iloc[:,0], samples_values_arr.iloc[:,1], s=10 * (1 - target_feature_arr),
                label='good', marker='o')
    plt.xlabel(dataSet.dataSet.columns.values[0])
    plt.ylabel(dataSet.dataSet.columns.values[1])

    print(regressionLine)
    plt.plot(samples_values_arr.iloc[:,0], regressionLine, color='r')
    plt.title(dataSet.data_set_name + ' data set')
    # print(logReg.square_error)
    #
    plt.figure(2)
    plt.plot(range(numofIteration), logReg.square_error)
    plt.xlabel('Num of iterations')
    plt.ylabel('Square Error')
    plt.title('Cost Function over iteration')

    plt.legend()
    plt.show()

