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
