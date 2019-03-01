import keras as K
from sklearn.metrics import confusion_matrix as cm
import numpy as np


class EvaluationMetrics(object):

    def jaccard_index(self, y_true, y_pred):
        """Implementation of intersection over union method for binary arrays.
        :param y_true: numpy.array
        :param y_pred: numpy.array
        :rtype: float
        """
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        area_of_intersection = sum((y_true & y_pred).flatten())
        area_of_union = sum((y_true | y_pred).flatten())
        return float(area_of_intersection) / area_of_union

    def precision(self, y_true, y_pred):
        """Implementation of the precision score.
       :param y_true: numpy.array
       :param y_pred: numpy.array
       :rtype: float
       """
        # Count positive samples.
        c1 = K.backend.sum(K.backend.round(K.backend.clip(y_true * y_pred, 0, 1)))
        c2 = K.backend.sum(K.backend.round(K.backend.clip(y_pred, 0, 1)))

        # How many selected items are relevant?
        return c1 / c2

    def recall(self, y_true, y_pred):
        """Implementation of the recall score.
        :param y_true: numpy.array
        :param y_pred: numpy.array
        :rtype: float
       """
        # Count positive samples.
        c1 = K.backend.sum(K.backend.round(K.backend.clip(y_true * y_pred, 0, 1)))
        c3 = K.backend.sum(K.backend.round(K.backend.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if c3 == 0:
            return 0
        # How many relevant items are selected?
        return c1 / c3

    def f1(self, y_true, y_pred):
        """Implementation of the f1 score, f1 is the harmonic average of the precision and recall.
        :param y_true: numpy.array
        :param y_pred: numpy.array
        :rtype: float
        """
        recall_score = self.recall(y_true, y_pred)
        precision_score = self.precision(y_true, y_pred)
        # Calculate f1_score
        return 2 * (precision_score * recall_score) / (precision_score + recall_score)

    def confusion_matrix(self, y_true, y_pred):
        """Implementation of the confusion matrix.
        :param y_true: numpy.array
        :param y_pred: numpy.array
        :rtype: numpy.array
        """

        return cm(y_true, y_pred)

    def normalised_confusion_matrix(self, y_true, y_pred):
        """Implementation of the normalised confusion matrix.
        :param y_true: numpy.array
        :param y_pred: numpy.array
        :rtype: numpy.array
        """
        conf_mat =  self.confusion_matrix(y_true, y_pred)
        return conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]