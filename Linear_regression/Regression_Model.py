import numpy as np
import Supp_Algorithms as supp
from scipy import stats as stat

class Regression:
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
        # Normalize the features
        self.norm_X, self.stat_X = supp.DataPreparation.featuresNormaliz(self.X)

        self.W = np.array([])
        self.square_error = []

    def decayLearningRate(self, alpha0, decayingConst, iter_mum):
        """
        use learning rate decay, which allows the learning rate to start at a large value and then decay over time

        :param alpha0: the initial learning rate
        :param decayingConst: controls how quickly the learning rate decays.
        :param iter_mum: the current gradient descent iteration number
        :return: the updated learning rate
        """

        return alpha0 * (decayingConst / (decayingConst + iter_mum))


class LinearRegression(Regression):
    """
    LinearRegression: is a class for the Linear Regression algorithm

    @TODO: Need to handling categorical features.
    @TODO: Using Panda to facilitate the data processing
    The Algorithm Expansions:
                - Some important observation we can get from the trained model weights:
                    * The sign and the magnitude of the weights indicate the descriptive features have a positive and
                      a negative impact on the predication, ex the positive weight with 0.6, indicate that with
                      the increasing in this particular feature the target feature with 0.6,
                      the same regarding to the negative value it will lead to decrease the target feature.
                    * We can apply the statistical significance test to determine
                      the importance of each descriptive feature in the model. By applying the Null hypothesis and then
                      determined if there is enough evidence to accept or reject this hypothesis.
                        To Calculate the statistical significance Test:
                         1- calculate the standard error for overall system:
                            SE = squrt( Delta Error Squared / n - 2), where n is the number of instance in the data set
                         2- Calculate the standard error for the descriptive feature:
                            SE(X[j]) =  SE / (Squrt( Sum(Xi[j] - X_mean[j])^2 ))
                         3- Calculate the t-statistic value:
                            t =  W[j] / SE(X[j])
                         Unsig standard t-statistic look-up table, we can then determine the p-value
                         If the p-value is less than the required significance level, typically 0.05, we
                         reject the null hypothesis and say that the descriptive feature has a significant impact on
                         the model.


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
        super().__init__(samples_values_arr, target_values)

    def gradientDescent(self, initlearningRate, numofIteration, randA=-2, randB=2, adaptLearningRate=True,
                        decayingConst = 100):
        """
            Apply the gradient descent algorithm to create a linear predication model
            steps:
                1- generate random weights based on the RandA and RandB
                DONE-->> "We should store the mean and the stander deviation in our model to be use in the system predication"
                2- Normalize the descriptive features
                3- Repeat till number of iteration:
                    3.1 calculate the delta error by using the current model weights
                    3.2 calculate the delta error derivative by multiple in the descriptive features
                    3.3 Update the model weights by subtract the old weight from the delta error first derivative after
                        multiple it in the learning rate
                    3.4 store the square error value for the new model weight to make sure that the error is reducing
                5- calculate the predication values with the generated model

        :param initlearningRate: is the learning step, which use to change the weights to lead the model to converge
                if the learningRate is very low, it will need mush more iteration to converge
                if the LearningRate is very large, The large adjustments made to the weights during gradient descent
                 cause it to jump completely from one side of the error surface to the other.
                 Although the algorithm can still converge toward an area of the error surface close to the global minimum,
                  there is a strong chance that the global minimum itself will actually be missed,
                  and the algorithm will simply jump back and forth across it.
        :param numofIteration: number of iteration the the gradient descent will iterate to build the linear  model
        :param randA: the start of the random range which use to generate a random weights
        :param randB: the end of the random range which use to generate a random weights
        :param adaptLearningRate: Giving the option to apply the learning rate decaying during the learning process
        :param decayingConst: controls how quickly the learning rate decays.
        :return: the function return two information:
                        1- The first output is the model weights after the converge
                        2- The second output is the predication target values
                        3- Save the mean and the standard deviation for the descriptive features
        """
        row, col = self.X.shape
        m = row

        # [a, b), b > a multiply the output of random_sample by (b-a) and add a:
        self.W = (randB - randA) * np.random.random((1, col)) + randA
        learningRate = initlearningRate


        for it in range(numofIteration):

            deltaError = supp.CostFunction.DeltaError(self.W, self.norm_X, self.Y)
            deltaError_drv = (1 / m) * (deltaError.transpose() @ self.norm_X)

            if adaptLearningRate:
                learningRate = self.decayLearningRate(initlearningRate, decayingConst, it)

            self.W = self.W - learningRate * deltaError_drv

            cost = supp.CostFunction.DeltaError(self.W, self.norm_X, self.Y)
            cost = np.power(cost, 2)
            sum_cost = np.sum(cost) / 2 * m
            self.square_error.append(sum_cost)

        regressionLine = self.norm_X @ np.transpose(self.W)
        return self.W, regressionLine

    def stat_significance_calc(self, alpha = 0.05):
        """
        We can apply the statistical significance test to determine the importance of each descriptive
        feature in the model. By applying the Null hypothesis and then
        determined if there is enough evidence to accept or reject this hypothesis.
            To Calculate the statistical significance Test:
                1- calculate the standard error for overall system:
                    SE = squrt( Delta Error Squared / n - 2), where n is the number of instance in the data set
                2- Calculate the standard error for the descriptive feature:
                    SE(X[j]) =  SE / (Squrt( Sum(Xi[j] - X_mean[j])^2 ))
                3- Calculate the t-statistic value:
                    t =  W[j] / SE(X[j])
        Unsig standard t-statistic look-up table, we can then determine the p-value
        If the p-value is less than the required significance level, typically 0.05, we reject the null hypothesis
        and say that the descriptive feature has a significant impact on the model.

        :param alpha: predefined significance threshold, and if the p-value is less than or equal to the threshold
        (i.e., the p-value is small), the null hypothesis is rejected. These thresholds are typically
         the standard statistical thresholds of 5% or 1%.
        :return: t_statistic: an array of the t-statistic values for the descriptive features.
                 p_values: an array of the p-values for the t-statistic values
                 null_hypothesis: an array contain an conclusion for the significance test.
        """
        row, col = self.norm_X.shape
        #df : degree of freedom
        df = row - 2
        deltaError = supp.CostFunction.DeltaError(self.W, self.norm_X, self.Y)
        deltaErrorsqu = np.power(deltaError, 2)
        deltaErrorsqu = np.sum(deltaErrorsqu)
        deltaErrorsqu = deltaErrorsqu / (row - 2)
        SE = np.sqrt(deltaErrorsqu)

        t_statistic = np.zeros(shape=(1,col))
        p_values = np.zeros(shape=(1,col))
        null_hypothesis = [None]

        t_statistic[0, 0] = None
        p_values[0, 0] = None


        # cv = stat.t.ppf(1.0 - alpha, df)
        # print(cv)

        for colidx in range(1,col):
            SE_Xi = self.X[:, colidx] - self.stat_X[0, colidx]
            SE_Xi = np.power(SE_Xi, 2)
            SE_Xi = np.sqrt(sum(SE_Xi))
            SE_Xi = SE / SE_Xi
            t_statistic[0, colidx] = self.W[0, colidx] / SE_Xi
            p_values[0, colidx] = (1 - stat.t.cdf(abs(t_statistic[0, colidx]), df)) * 2.0
            if p_values[0, colidx] <= alpha:
                null_hypothesis.append('Have a significant impact')
            else:
                null_hypothesis.append('Dose not have a significant impact')


        return t_statistic, p_values, null_hypothesis

class LogisticRegression(Regression):

    def __init__(self, samples_values_arr, target_values):
        """
        The Logistic Algorithm initialization

        :param samples_values_arr: the sample values array "it could be a 2D list the convert happen inside the function"
        :param target_values: the target value vector "it could be a 1D list the convert happen inside the function"
        """

        super().__init__(samples_values_arr, target_values)

    def gradientDescent(self, initlearningRate, numofIteration, randA=-2, randB=2, adaptLearningRate=True,
                        decayingConst = 100):
        """
        Applying the gradient descent for logistic regression to classify between two category of target feature

        :param initlearningRate: is the learning step, which use to change the weights to lead the model to converge
                if the learningRate is very low, it will need mush more iteration to converge
                if the LearningRate is very large, The large adjustments made to the weights during gradient descent
                 cause it to jump completely from one side of the error surface to the other.
                 Although the algorithm can still converge toward an area of the error surface close to the global minimum,
                  there is a strong chance that the global minimum itself will actually be missed,
                  and the algorithm will simply jump back and forth across it.
        :param numofIteration: number of iteration the the gradient descent will iterate to build the linear  model
        :param randA: the start of the random range which use to generate a random weights
        :param randB: the end of the random range which use to generate a random weights
        :param adaptLearningRate: Giving the option to apply the learning rate decaying during the learning process
        :param decayingConst: controls how quickly the learning rate decays.
        :return: the function return two information:
                        1- The first output is the model weights after the converge
                        2- The second output is the decision boundary
        """

        row, col = self.X.shape
        m = row

        self.W = (randB - randA) * np.random.random((1, col)) + randA
        learningRate = initlearningRate

        for it in range(numofIteration):

            # (N X M) @ (M X 1) = (N X 1)
            predicate = self.norm_X @ np.transpose(self.W)
            predicate = supp.LogisticFuction(predicate)
            # Y -> (N X 1) - predicate -> (N X 1) = (N X 1)
            diff_val = (predicate - self.Y)

            # (1 X N) @ (N X M)
            diff_val = np.transpose(diff_val) @ self.norm_X  # (1 X M)
            diff_val = diff_val / m

            if adaptLearningRate:
                learningRate = self.decayLearningRate(initlearningRate, decayingConst, it)

            self.W = self.W - learningRate * diff_val

            cost = supp.CostFunction.crossEntropy(supp.LogisticFuction(self.norm_X @ np.transpose(self.W)), self.Y)

            self.square_error.append(cost)

        # regression_output = supp.CostFunction.LogisticFuction(self.W, self.norm_X)
        # To Draw the boundary decision we need to draw the line that satisfy that
        #       p(Y = 1| X) = p(Y = 0| x) = 1/2
        #       1/2 = (1 / 1 + exp(-1 * W @ X) )     If we simplify this equation
        #       2 = 1 + exp(-1 * W @ X)
        #       1 = exp(-1 * W @ X) take the natural log
        #       0 = -1 * W @ X
        #       W @ X = 0
        #   so if we have only two feature
        #       w0 + w1*x1 + w2 * x2 = 0
        #       x2 = - (w1 / w2) x1 - (w0 / w2)
        #       X2 = (-1 / w2) * (w1 * x1 + w0)

        w0 = self.W[0, 0]
        w1 = self.W[0, 1]
        w2 = self.W[0, 2]
        regression_output = (-1.0 / w2) * (w1 * self.norm_X[:, 1] + w0)
        regression_output = regression_output * self.stat_X[1,2] + self.stat_X[0,2]

        return self.W, regression_output



