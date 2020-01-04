import numpy as np

class CostFunction:

    @staticmethod
    def DeltaError(W, X, Y):
        """
        Square Error Function compute the error between the predication model and the target feature
        based on the equation:
                squE = (( (W)T * X) - Y)^2
        :param W: The weights Vector
        :param X: The descriptive features vector, it should contain the dummy Feature x0
        :param Y: The target features Vector
        :return: a vector of the square error between the predicated targets and the actual target
        """

        W_trans = np.array(W)
        predicate = X @ W_trans.transpose()
        deltaError = predicate.transpose() - Y

        return deltaError

class DataPreparation:

    @staticmethod
    def featureNormalize(Xi):
        """
        This function normalize a vector of feature
        :param Xi: a vector contain the feature value
        :return: normalized the feature vector
        """

        return (Xi - np.mean(Xi)) / np.std(Xi)

    @staticmethod
    def featuresNormaliz(X):
        """
        featuresNormaliz: is a function to normalize a data set descriptive features
        :param X: Array of descriptive features
        :return: Normalized descriptive features array
        """
        row, col = X.shape
        norm_X = np.array(X)
        for colIdx in range(col, 1):
            norm_X[:, colIdx] = DataPreparation.featureNormalize(X[:, colIdx])

        return norm_X


