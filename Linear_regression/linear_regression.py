import numpy as np
import Supp_Algorithms as supp


class LinearRegression:
    """
    LinearRegression: is a class for the Linear Regression algorithm

    self.X : is the sample values array after adding the dummy feature
    self.Y : is the target value vector
    self.W : is the Weight of our linear model
    """

    def __init__(self, samples_values_arr, target_values):
        """
        The linear Algorithm initialization
        :param samples_values_arr: the sample values array "it could be a 2D list the convert happen inside the function"
        :param target_values: the target value vector "it could be a 1D list the convert happen inside the function"

        """
        self.X = np.array(samples_values_arr)
        self.Y = np.array(target_values)
        # add the dummy feature at rhe beginning
        self.X = np.insert(self.X, 0, 1, axis=1)
        self.W = np.array([])
        self.square_error = []

    def gradientDescent(self, learningRate, numofIteration, randA=-2, randB=2):
        """
            Apply the gradient descent algorithm to create a linear predication model
            steps:
                1- generate random weights based on the RandA and RandB
                @Todo "We should store the mean and the stander deviation in our model to be use in the system predication"
                2- Normalize the descriptive features
                3- Repeat till number of iteration:
                    3.1 calculate the delta error by using the current model weights
                    3.2 calculate the delta error derivative by multiple in the descriptive features
                    3.3 Update the model weights by subtract the old weight from the delta error first derivative after
                        multiple it in the learning rate
                    3.4 store the square error value for the new model weight to make sure that the error is reducing
                5- calculate the predication values with the generated model

        :param learningRate: is the learning step, which use to change the weights to lead the model to converge
                if the learningRate is very low, it will need mush more iteration to converge
                if the LearningRate is very large, The large adjustments made to the weights during gradient descent
                 cause it to jump completely from one side of the error surface to the other.
                 Although the algorithm can still converge toward an area of the error surface close to the global minimum,
                  there is a strong chance that the global minimum itself will actually be missed,
                  and the algorithm will simply jump back and forth across it.
        :param numofIteration: number of iteration the the gradient descent will iterate to build the linear  model
        :param randA: the start of the random range which use to generate a random weights
        :param randB: the end of the random range which use to generate a random weights
        :return: the function return two information:
                        1- The first output is the model weights after the converge
                        2- The second output is the predication target values
                        3- Save the mean and the standard deviation for the descriptive features
        """
        row, col = self.X.shape
        m = row

        # [a, b), b > a multiply the output of random_sample by (b-a) and add a:
        self.W = (randB - randA) * np.random.random((1, col)) + randA
        # Normalize the features
        norm_X, self.stat_X = supp.DataPreparation.featuresNormaliz(self.X)


        for it in range(numofIteration):

            deltaError = supp.CostFunction.DeltaError(self.W, norm_X, self.Y)
            deltaError_drv = (1 / m) * (deltaError @ norm_X)
            self.W = self.W - learningRate * deltaError_drv

            cost = supp.CostFunction.DeltaError(self.W, norm_X, self.Y)
            cost = np.power(cost, 2)
            sum_cost = np.sum(cost) / 2 * m
            self.square_error.append(sum_cost)

        regressionLine = norm_X @ np.transpose(self.W)
        return self.W, regressionLine
