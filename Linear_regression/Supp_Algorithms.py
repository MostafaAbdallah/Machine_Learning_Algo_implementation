import numpy as np


def LogisticFuction(Val):
    """
    Function to calculate the logistic cost
    :param Val: an array, a vector or scalar to calculate the logistic function to it
    :return: logistic fun values
    """

    Val = -1.0 * Val

    logistc_val = 1.0 / (1.0 + np.exp(Val))

    return logistc_val


class CostFunction:

    @staticmethod
    def DeltaError(W, X, Y):
        """
        Error Function compute the error between the predication model and the target feature
        based on the equation:
                DE = (( (W)T * X) - Y)
        :param W: The weights Vector
        :param X: The descriptive features vector, it should contain the dummy Feature x0
        :param Y: The target features Vector
        :return: a vector of the delta error between the predicated targets and the actual target
        """

        W_trans = np.array(W)
        predicate = X @ W_trans.transpose()
        deltaError = predicate - Y

        return deltaError

    @staticmethod
    def crossEntropy(predication, Y):
        """

        :param predication: (N X 1)
        :param Y: (N X 1)
        :return: cost as scale value
        """

        row, col = Y.shape
        # cost _scalar = (1 X N) * (N X 1)
        class1 = Y * np.log(predication)
        class2 = (1 - Y) * np.log((1-predication))
        cost = class1 + class2
        cost = sum(cost) * -1/row

        return float(cost)


class DataPreparation:

    @staticmethod
    def featureNormalize(Xi):
        """
        This function normalize a vector of feature
        :param Xi: a vector contain the feature value
        :return: normalized the feature vector
        """
        x_mean = np.mean(Xi)
        x_std = np.std(Xi)
        #
        return (Xi - x_mean) / x_std, np.array([x_mean, x_std])

    @staticmethod
    def featuresNormaliz(X):
        """
        featuresNormaliz: is a function to normalize a data set descriptive features
        :param X: Array of descriptive features
        :return: Normalized descriptive features array
        """
        row, col = X.shape
        norm_X = np.array(X, dtype=float)
        stat_X = np.zeros(shape=(2, col))
        for colIdx in range(1, col):
            norm_X[:, colIdx], tmp = DataPreparation.featureNormalize(X[:, colIdx])
            stat_X[:, colIdx] = tmp

        return norm_X, stat_X


