from sklearn.base import ClassifierMixin
from sklearn.ensemble import VotingClassifier
from .batch_generator import batch_generator
from pandas.core.frame import DataFrame
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelRunner:

    def train(self, no_epochs: int):
        pass

    def test(self):
        pass

    def plot_results(self):
        pass


class cluster_runner(ModelRunner):

    def __init__(self, models: [(str, ClassifierMixin)], x_train: DataFrame, y_train: DataFrame, x_test: DataFrame,
                 y_test: DataFrame):
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cluster = self.cluster = VotingClassifier(estimators=self.models, voting="hard")
        self.results = dict.fromkeys(["confusionmatrices", "accuracies", "features", "avgconf", "avgacc"])

    def train(self, no_epochs: int, plot_intermediate: bool = False):
        confusion_matrices = []
        accuracies = []

        for epoch in range(no_epochs):
            train_batch_x, train_batch_y = \
                batch_generator.get_batch(batch_size=256, data=self.x_train, labels=self.y_train)

            self.cluster = self.cluster.fit(train_batch_x, train_batch_y.to_numpy().ravel())

            test_batch_x, test_batch_y = \
                batch_generator.get_batch(batch_size=256, data=self.x_test, labels=self.y_test)

            y_pred = self.cluster.predict(test_batch_x)
            confusion_matrices.append(confusion_matrix(test_batch_y, y_pred))
            accuracies.append(accuracy_score(test_batch_y, y_pred))

        avgconf = sum(confusion_matrices) / no_epochs
        avgaccuracies = sum(accuracies) / no_epochs

        #update object result attribute:
        self.results["confusionmatrices"] = confusion_matrices
        self.results["accuracies"] = accuracies
        self.results["avgconf"] = avgconf
        self.results["avgacc"] = avgaccuracies





    def test(self):
        pass

    def plot_results(self):
        pass


class ensemble_runner(ModelRunner):
    def __init__(self, clusters: [cluster_runner]):
        self.clusters = clusters
        self.ensemble = None
        self.results = dict()

    def train(self, no_epochs: int):
        pass

    def test(self):
        pass

    def plot_results(self):
        pass