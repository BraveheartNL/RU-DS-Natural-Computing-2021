from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn import metrics
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


# Assignment 5.5
def ada_boost_experiment():
    '''
    Divided subsections of experimentation with the Scikit-learn basic AdaboostClassifier.
    Results in png output of graphs containing visual representation of toy dataset
    Ada classifier decision boundaries and test/train average precision,
    mean error and recall scores.
    Experimented factors are:
     - number of data points (n)
     - number of classifier estimators
     - classifier type (standard is DecisionTreeClassifier)
    :return: Nothing
    '''

    # -----------------------------------------------------------------------------------------------------------------
    # Experiment 1: Number of Estimators = 2, dataset size = 26, classifier = DecisionTreeClassifier
    # -----------------------------------------------------------------------------------------------------------------

    title = "AdaBoost_ne2_dsz26_DTC"
    x, y = make_toy_dataset(n=26, random_seed=10)

    # Split the dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    classifier = AdaBoostClassifier(n_estimators=2, algorithm='SAMME')

    model = classifier.fit(x_train, y_train)

    # Plot evaluation metrics and decision boundaries on train and test set:
    plot_adaboost(x_train, y_train, model, output=True, title=title)
    plot_adaboost(x_test, y_test, model, test=True, output=True, title=title)

    # -----------------------------------------------------------------------------------------------------------------
    # Experiment 2: Number of Estimators = 4, dataset size = 26, classifier = DecisionTreeClassifier
    # -----------------------------------------------------------------------------------------------------------------

    title = "AdaBoost_ne4_dsz26_DTC"
    x, y = make_toy_dataset(n=26, random_seed=10)

    # Split the dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    classifier = AdaBoostClassifier(n_estimators=4, algorithm='SAMME')

    model = classifier.fit(x_train, y_train)

    # Plot evaluation metrics and decision boundaries on train and test set:
    plot_adaboost(x_train, y_train, model, output=True, title=title)
    plot_adaboost(x_test, y_test, model, test=True, output=True, title=title)

    # -----------------------------------------------------------------------------------------------------------------
    # Experiment 3: Number of Estimators = 6, dataset size = 26, classifier = DecisionTreeClassifier
    # -----------------------------------------------------------------------------------------------------------------

    title = "AdaBoost_ne6_dsz26_DTC"
    x, y = make_toy_dataset(n=26, random_seed=10)

    # Split the dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    classifier = AdaBoostClassifier(n_estimators=6, algorithm='SAMME')

    model = classifier.fit(x_train, y_train)

    # Plot evaluation metrics and decision boundaries on train and test set:
    plot_adaboost(x_train, y_train, model, output=True, title=title)
    plot_adaboost(x_test, y_test, model, test=True, output=True, title=title)

    # -----------------------------------------------------------------------------------------------------------------
    # Experiment 4: Number of Estimators = 8, dataset size = 26, classifier = DecisionTreeClassifier
    # -----------------------------------------------------------------------------------------------------------------

    title = "AdaBoost_ne8_dsz26_DTC"
    x, y = make_toy_dataset(n=26, random_seed=10)

    # Split the dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    classifier = AdaBoostClassifier(n_estimators=8, algorithm='SAMME')

    model = classifier.fit(x_train, y_train)

    # Plot evaluation metrics and decision boundaries on train and test set:
    plot_adaboost(x_train, y_train, model, output=True, title=title)
    plot_adaboost(x_test, y_test, model, test=True, output=True, title=title)

    # -----------------------------------------------------------------------------------------------------------------
    # Experiment 5: Number of Estimators = 10, dataset size = 26, classifier = DecisionTreeClassifier
    # -----------------------------------------------------------------------------------------------------------------

    title = "AdaBoost_ne10_dsz26_DTC"
    x, y = make_toy_dataset(n=26, random_seed=10)

    # Split the dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    classifier = AdaBoostClassifier(n_estimators=10, algorithm='SAMME')

    model = classifier.fit(x_train, y_train)

    # Plot evaluation metrics and decision boundaries on train and test set:
    plot_adaboost(x_train, y_train, model, output=True, title=title)
    plot_adaboost(x_test, y_test, model, test=True, output=True, title=title)

    # -----------------------------------------------------------------------------------------------------------------
    # Experiment 6: Number of Estimators = 10, dataset size = 260, classifier = DecisionTreeClassifier
    # -----------------------------------------------------------------------------------------------------------------

    title = "AdaBoost_ne10_dsz260_DTC"
    x, y = make_toy_dataset(n=260, random_seed=10)

    # Split the dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    classifier = AdaBoostClassifier(n_estimators=10, algorithm='SAMME')

    model = classifier.fit(x_train, y_train)

    # Plot evaluation metrics and decision boundaries on train and test set:
    plot_adaboost(x_train, y_train, model, output=True, title=title)
    plot_adaboost(x_test, y_test, model, test=True, output=True, title=title)

    # -----------------------------------------------------------------------------------------------------------------
    # Experiment 7: Number of Estimators = 10, dataset size = 2600, classifier = DecisionTreeClassifier
    # -----------------------------------------------------------------------------------------------------------------

    title = "AdaBoost_ne10_dsz2600_DTC"
    x, y = make_toy_dataset(n=2600, random_seed=10)

    # Split the dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    classifier = AdaBoostClassifier(n_estimators=10, algorithm='SAMME')

    model = classifier.fit(x_train, y_train)

    # Plot evaluation metrics and decision boundaries on train and test set:
    plot_adaboost(x_train, y_train, model, output=True, title=title)
    plot_adaboost(x_test, y_test, model, test=True, output=True, title=title)


# Adapted from: https://geoffruddock.com/adaboost-from-scratch-in-python/
# Credits to Geoff Ruddock, March 2020.
def make_toy_dataset(n: int = 100, random_seed: int = None):
    """ Generate a toy dataset for evaluating AdaBoost classifiers """

    if random_seed:
        np.random.seed(random_seed)

    x, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2)

    return x, y * 2 - 1


def plot_adaboost(X: np.ndarray,
                  y: np.ndarray,
                  clf=None,
                  sample_weights: Optional[np.ndarray] = None,
                  annotate: bool = False,
                  test: bool = False,
                  title: str = "AdaBoost dataset plot",
                  output: bool = False,
                  ax: Optional[mpl.axes.Axes] = None) -> None:
    """ Plot ± samples in 2D, optionally with decision boundary """

    assert set(y) == {-1, 1}, 'Expecting response labels to be ±1'

    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.set_facecolor('white')
        fig.suptitle(title)

    pad = 1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    if sample_weights is not None:
        sizes = np.array(sample_weights) * X.shape[0] * 100
    else:
        sizes = np.ones(shape=X.shape[0]) * 100

    X_pos = X[y == 1]
    sizes_pos = sizes[y == 1]
    ax.scatter(*X_pos.T, s=sizes_pos, marker='+', color='red')

    X_neg = X[y == -1]
    sizes_neg = sizes[y == -1]
    ax.scatter(*X_neg.T, s=sizes_neg, marker='.', c='blue')

    if clf:
        plot_step = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # If all predictions are positive class, adjust color map acordingly
        if list(np.unique(Z)) == [1]:
            fill_colors = ['r']
        else:
            fill_colors = ['b', 'r']

        ax.contourf(xx, yy, Z, colors=fill_colors, alpha=0.2)

        y_pred = clf.predict(X)
        mean_err = (y_pred != y).mean()
        avg_prec = metrics.average_precision_score(y, y_pred)
        recall = metrics.recall_score(y, y_pred)

        if test:
            mean_acc_str = f"Mean Test Error = {mean_err:0.1%}, " \
                           f"Average Test Precision = {avg_prec:0.1%}, " \
                           f"Test Recall = {recall:0.1%}."
        else:
            mean_acc_str = f"Mean Train Error = {mean_err:0.1%}, " \
                           f"Average Train Precision = {avg_prec:0.1%}, " \
                           f"Train Recall = {recall:0.1%}."

        plt.figtext(0.05, 0.01, mean_acc_str, fontsize=8)
        if not output:
            plt.show()

    if annotate:
        for i, (x, y) in enumerate(X):
            offset = 0.05
            ax.annotate(f'$x_{i + 1}$', (x + offset, y - offset))

    ax.set_xlim(x_min + 0.5, x_max - 0.5)
    ax.set_ylim(y_min + 0.5, y_max - 0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    if not output:
        plt.show()
    else:
        output_path = "{}/output".format(os.getcwd())
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        output = '{}_{}.png'.format(title.replace(" ", "_"), "test" if test else "train")
        fig.savefig(fname=os.path.join(output_path, output))


if __name__ == '__main__':
    ada_boost_experiment()
