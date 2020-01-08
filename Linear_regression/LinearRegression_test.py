import read_dataset
from Regression_Model import LinearRegression
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def Test1():
    dataSet = read_dataset.DataSet('./DataSet/franchise_rest.csv')
    learningRate = 0.5
    numofIteration = 100
    samples_values_arr, target_feature_arr = dataSet.getDataSetAsNumPy()

    linearReg = LinearRegression(samples_values_arr, target_feature_arr)
    model_weights, regressionLine = linearReg.gradientDescent(initlearningRate=learningRate, numofIteration=numofIteration)

    plt.figure(1)

    plt.scatter(samples_values_arr, target_feature_arr, label=dataSet.data_set_name, marker='x')
    plt.xlabel(dataSet.feature_name[0])
    plt.ylabel(dataSet.target_name)

    plt.plot(samples_values_arr, regressionLine)
    plt.title(dataSet.data_set_name + ' data set')
    print(linearReg.square_error)

    plt.figure(2)
    plt.plot(range(numofIteration), linearReg.square_error)
    plt.xlabel('Num of iterations')
    plt.ylabel('Square Error')
    plt.title('Cost Function over iteration')

    plt.legend()
    plt.show()


def Test2():
    dataSet = read_dataset.DataSet('./DataSet/rental_house.csv')

    learningRate = 0.1
    numofIteration = 100
    samples_values_arr, target_feature_arr = dataSet.getDataSetAsNumPy()

    linearReg = LinearRegression(samples_values_arr, target_feature_arr)
    model_weights, regressionLine = linearReg.gradientDescent(initlearningRate=learningRate, numofIteration=numofIteration)

    plt.figure(1)

    plt.scatter(samples_values_arr, target_feature_arr, label=dataSet.data_set_name, marker='x')
    plt.xlabel(dataSet.feature_name[0])
    plt.ylabel(dataSet.target_name)

    plt.plot(samples_values_arr, regressionLine)
    plt.title(dataSet.data_set_name + ' data set')
    print(linearReg.square_error)

    plt.figure(2)
    plt.plot(range(numofIteration), linearReg.square_error)
    plt.xlabel('Num of iterations')
    plt.ylabel('Square Error')
    plt.title('Cost Function over iteration')

    plt.legend()
    plt.show()


def Test3():
    dataSet = read_dataset.DataSet('./DataSet/House_Price.csv')
    learningRate = 1.5
    numofIteration = 100
    samples_values_arr, target_feature_arr = dataSet.getDataSetAsNumPy()

    linearReg = LinearRegression(samples_values_arr, target_feature_arr)
    model_weights, regressionLine = linearReg.gradientDescent(initlearningRate=learningRate, numofIteration=numofIteration)

    t_stat, p_values, null_hypothesis = linearReg.stat_significance_calc()
    print('t-statistic :')
    print(t_stat)
    print('P_values :')
    print(p_values)
    print('significant impact')
    print(null_hypothesis)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter3D(samples_values_arr[:, 0], samples_values_arr[:, 1], np.transpose(target_feature_arr),
                    marker='x', label="The actual Values")
    ax.scatter3D(samples_values_arr[:, 0], samples_values_arr[:, 1], regressionLine,
                    marker='o', label="The predicated Values")

    ax.set_xlabel(dataSet.feature_name[0])
    ax.set_ylabel(dataSet.feature_name[1])
    ax.set_zlabel(dataSet.target_name)

    plt.title(dataSet.data_set_name + ' data set')
    print(linearReg.square_error)
    # #
    plt.figure(2)
    plt.plot(range(numofIteration), linearReg.square_error)
    plt.xlabel('Num of iterations')
    plt.ylabel('Square Error')
    plt.title('Cost Function over iteration')
    #
    # plt.legend()
    plt.show()
