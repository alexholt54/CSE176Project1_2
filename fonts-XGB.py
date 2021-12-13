# Use fonts dataset
from sklearn.utils import validation
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
import os
import glob
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

def main():

    df = pd.read_csv("datasets/allFont.csv")

    labels = df["font"].values
    df = df.drop(['font', 'Unnamed: 0'], axis=1)
    data = df.to_numpy()

    labels = labels.flatten()

    for i in range(len(labels)):
        if labels[i] == 'GOUDY':
            labels[i] = 0
        elif labels[i] == 'NIAGARA':
            labels[i] = 1
        elif labels[i] == 'ARIAL':
            labels[i] = 2
        elif labels[i] == 'CAMBRIA':
            labels[i] = 3
        elif labels[i] == 'COMIC':
            labels[i] = 4
        elif labels[i] == 'HARLOW':
            labels[i] = 5
        elif labels[i] == 'PAPYRUS':
            labels[i] = 6
        elif labels[i] == 'ROMAN':
            labels[i] = 7
        elif labels[i] == 'TIMES':
            labels[i] = 8
        elif labels[i] == 'MODERN':
            labels[i] = 9

    x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.33, random_state=0)

    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.4, random_state=0)

    """rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    valError = pd.DataFrame([], columns = ["rate", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["rate", "error"])

    for rate in rates:

        model = XGBClassifier(learning_rate = rate, use_label_encoder=False)
        model.fit(x_train, y_train)

        y_preds_train = model.predict(x_train)
        y_preds = model.predict(x_valid)

        numCorrect = 0
        for i in range(len(y_train)):
            if y_train[i] == y_preds_train[i]:
                numCorrect += 1
        train_error = 1 - (numCorrect / len(y_train))
        numCorrect = 0
        for i in range(len(y_valid)):
            if y_valid[i] == y_preds[i]:
                numCorrect += 1
        error = 1 - (numCorrect / len(y_valid))

        row1 = {"rate" : rate, "error" : error}
        row3 = {"rate" : rate, "error" : train_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)

    ax = valError.plot(x = "rate", y = "error", kind = "line", color = "red", label = "Validation")
    valErrorTrain.plot(x = "rate", y = "error", kind = "line", ax = ax, color = "blue", label = "Training",
                        title = "XGB Error With Varying Learning Rates (Fonts)", ylabel = "Error", xlabel = "Learning Rate")
    plt.show()"""

    """     trees = [1, 10, 250, 500, 1000]

    valError = pd.DataFrame([], columns = ["tree", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["tree", "error"])

    for tree in trees:
        model = XGBClassifier(n_estimators = tree, use_label_encoder=False)
        model.fit(x_train, y_train)

        y_preds_train = model.predict(x_train)
        y_preds = model.predict(x_valid)

        numCorrect = 0
        for i in range(len(y_train)):
            if y_train[i] == y_preds_train[i]:
                numCorrect += 1
        train_error = 1 - (numCorrect / len(y_train))
        numCorrect = 0
        for i in range(len(y_valid)):
            if y_valid[i] == y_preds[i]:
                numCorrect += 1
        error = 1 - (numCorrect / len(y_valid))

        row1 = {"tree" : tree, "error" : error}
        row3 = {"tree" : tree, "error" : train_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)

    ax = valError.plot(x = "tree", y = "error", kind = "line", color = "red", label = "Validation")
    valErrorTrain.plot(x = "tree", y = "error", kind = "line", ax = ax, color = "blue", label = "Training",
                        title = "XGB Error With Varying Number of Trees (Fonts)", ylabel = "Error", xlabel = "Number of Trees")
    plt.show() """

    depths = [1, 10, 250, 500]

    valError = pd.DataFrame([], columns = ["depths", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["depths", "error"])

    for depth in depths:
        model = XGBClassifier(n_estimators = 500, learning_rate = 0.01, max_depth = depth, use_label_encoder=False)
        model.fit(x_train, y_train)

        y_preds_train = model.predict(x_train)
        y_preds = model.predict(x_valid)

        numCorrect = 0
        for i in range(len(y_train)):
            if y_train[i] == y_preds_train[i]:
                numCorrect += 1
        train_error = 1 - (numCorrect / len(y_train))
        numCorrect = 0
        for i in range(len(y_valid)):
            if y_valid[i] == y_preds[i]:
                numCorrect += 1
        error = 1 - (numCorrect / len(y_valid))

        row1 = {"depths" : depth, "error" : error}
        row3 = {"depths" : depth, "error" : train_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)

    ax = valError.plot(x = "depths", y = "error", kind = "line", color = "red", label = "Validation")
    valErrorTrain.plot(x = "depths", y = "error", kind = "line", ax = ax, color = "blue", label = "Training",
                        title = "XGB Error With Varying Number of Max Depths (Fonts)", ylabel = "Error", xlabel = "Number of Depths")
    plt.show()

# Call this to save data file to your machine
def loadDataFile():
    path = os.getcwd()
    csv_files = glob.glob(os.path.join("datasets/fonts", "*.csv"))
    mainDF = pd.DataFrame()

    for f in csv_files:
        df = pd.read_csv(f)
        #print('Location:', f)
        print('File Name:', f.split("\\")[-1])
        df = df.drop(['orientation', 'm_top', 'm_left', 'originalH', 'originalW', 'h', 'w', 'fontVariant', 'm_label', 'strength', 'italic'], axis=1)
        mainDF = mainDF.append(df)

    mainDF.to_csv('allFont.csv')

def normalizeData(data):
    data = (data / 255)
    train_mean = data.mean(axis = 0)
    train_std = data.std(axis = 0)
    train_std[train_std == 0] = 1
    data -= train_mean
    data /= train_std
    return data



if __name__ == "__main__":
    main()