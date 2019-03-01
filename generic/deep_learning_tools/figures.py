import pylab as plt
import seaborn as sns


class Plotter(object):

    def plot_confustion_matrix(self, confusion_matrix, ticklabels, figsize=(12, 12)):
        """
        :type confusion_matrix: numpy.array
        :type ticklabels: list[str]
        :type figsize: tuple(int, int)
        :rtype: matplotlib.Figure
        """
        fig = plt.figure(figsize=figsize)
        sns.heatmap(confusion_matrix, linewidths=.5, cmap="YlGnBu", xticklabels=ticklabels,
                    annot=True, yticklabels=ticklabels, cbar=False)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig