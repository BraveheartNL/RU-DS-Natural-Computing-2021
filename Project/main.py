import os
import pandas as pd
import numpy as np
from Models.model_runners import ensemble_runner, cluster_runner
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def train_multiclassifyer_adaboost():
    """

    :return:
    """

    # Search and Load dataset:
    dataset_dir = None
    dataset_file_name = "ai4i2020.csv"
    print("Looking for dataset named: {} ...".format(dataset_file_name))
    found = False

    for root, dirs, files in os.walk(os.getcwd()):
        for name in files:
            if name == dataset_file_name:
                dataset_dir = os.path.join(root, name)
                found = True
                break
        if found:
            break

    x = y = None

    if not dataset_dir:
        raise FileNotFoundError("Could not find dataset directory in current working directory.")
    else:
        # manual parse of the headers, as pandas seems to have issues with parsing non chars like [, ] and spaces.
        try:
            with open(dataset_dir, encoding='utf8') as f:
                headers = f.readline().replace("\n", "").replace(" ", "_").replace("[", "").replace("]", "").split(",")
            dataset = pd.read_csv(dataset_dir, index_col=0, delimiter=',', header=None, skiprows=[0], names=headers)
            print("Data-set found and loaded successfully.")
        except:
            raise ImportError("Data set parsing unsuccessful!")

    # --- extract model parameters from data: ---
    # total number of instances
    print("No. of instance:", len(dataset))

    # missing values and amount of missing values
    print("Missing values:", dataset.isnull().values.any())
    print("Amount of missing values:", dataset.isnull().sum().sum())

    # check amount of failure
    print("Amount of machine failures:", dataset["Machine_failure"].sum())

    # data types
    print(dataset.dtypes)

    #create x_data and labels:
    y = dataset.loc[:, "Machine_failure"]
    x = dataset.drop(["Machine_failure"], axis=1)

    #create test and train sets:
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        print("Data set test and train split made successfully. \n")
    except:
        raise ValueError("Data set split into data and target labels unsuccessful!")


    # Define models and pass to ensemble_runner:
    # Decision Tree:

    models = [("DTC", DecisionTreeClassifier(max_depth=3))]

    cluster = cluster_runner(models, x_train, y_train, x_test, y_test)
    pass

    # tuned_parameters = {
    #     'base_estimator': base_models,
    #     'loss': ['exponential'],
    #     'random_state': [47],
    #     'learning_rate': [1]
    # }
    cluster1_clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

    for clf, label in zip([clf1, clf2, clf3, cluster1_clf], ['SVC', 'KNN', 'DTC', 'Ensemble']):
        scores = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


if __name__ == '__main__':
    train_multiclassifyer_adaboost()
