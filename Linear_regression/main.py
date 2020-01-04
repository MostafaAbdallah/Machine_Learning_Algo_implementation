import read_dataset
from linear_regression import LinearRegression
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def Test1():
    dataSet = read_dataset.DataSet('./DataSet/franchise_rest.csv')
    learningRate = 0.01
    numofIteration = 1500
    samples_values_arr, target_feature_arr = dataSet.getDataSetAsNumPy()

    linearReg = LinearRegression(samples_values_arr, target_feature_arr)
    model_weights, regressionLine = linearReg.gradientDescent(learningRate=learningRate, numofIteration=numofIteration)

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

    learningRate = 0.000001
    numofIteration = 100
    samples_values_arr, target_feature_arr = dataSet.getDataSetAsNumPy()

    linearReg = LinearRegression(samples_values_arr, target_feature_arr)
    model_weights, regressionLine = linearReg.gradientDescent(learningRate=learningRate, numofIteration=numofIteration)

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
    learningRate = 0.00000001
    numofIteration = 1000
    samples_values_arr, target_feature_arr = dataSet.getDataSetAsNumPy()

    linearReg = LinearRegression(samples_values_arr, target_feature_arr)
    model_weights, regressionLine = linearReg.gradientDescent(learningRate=learningRate, numofIteration=numofIteration)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter3D(samples_values_arr[:, 0], samples_values_arr[:, 1], np.transpose(target_feature_arr),
                    marker='x')
    ax.scatter3D(samples_values_arr[:, 0], samples_values_arr[:, 1], regressionLine,
                    marker='o')

    ax.set_xlabel(dataSet.feature_name[0])
    ax.set_ylabel(dataSet.feature_name[1])
    ax.set_zlabel(dataSet.target_name)
    # plt.xlabel(dataSet.feature_name[0])
    # plt.ylabel(dataSet.target_name)
    #
    # plt.plot(samples_values_arr, regressionLine)
    # plt.title(dataSet.data_set_name + ' data set')
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


if __name__ == '__main__':
    # Test1()
    # Test2()
    Test3()
