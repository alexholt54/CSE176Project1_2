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

    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

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
    plt.show()
    
    quit()
    
    #Change Paramters here:
        
    minTrees = 100
    maxTrees = 1000

    trees = list(range(minTrees, maxTrees, 100))

    valError = pd.DataFrame([], columns = ["trees", "error"])

    for tree in trees:
        model = xgb.XGBClassifier(n_estimators = tree, use_label_encoder=False)
        model.fit(trainingSet, trainingLabels)

        row = {"trees" : tree, "error" : 1 - model.score(validationSet, validationLabels)}

        valError = valError.append(row, ignore_index=True)

        print(tree)

    ax = valError.plot(x = "trees", y = "error", kind = "line", color = "red", label = "Pixel Features",
                        title = "Validation Error With Varying Number of Trees", ylabel = "Validation Error", xlabel = "Number of Trees")
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