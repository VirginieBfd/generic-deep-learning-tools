import pylab as plt
import seaborn as sns


class Plotter(object):

    def plot_confusion_matrix(self, confusion_matrix, ticklabels, figsize=(12, 12)):
        """
        :type confusion_matrix: numpy.array
        :type ticklabels: list[str]
        :type figsize: tuple(int, int)
        :rtype: matplotlib.Figure
        """
        fig = plt.figure(figsize=figsize)
        sns.heatmap(
            confusion_matrix,
            cmap="YlGnBu",
            xticklabels=ticklabels,
            annot=True,
            yticklabels=ticklabels,
            cbar=False,
            fmt=".2f",
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        return fig
