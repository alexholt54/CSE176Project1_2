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

    # Data reading will go here...
    # 200 images from each file ~30k
    #70 for validation/testing

    df = pd.read_csv("datasets/allFont.csv")

    #print(np.shape(df))

    labels = df["font"].values
    df = df.drop(['font', 'Unnamed: 0'], axis=1)

    
    
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


    x_train, x_valid2, y_train, y_valid2 = train_test_split(df, labels, test_size=0.33, random_state=0)

    x_valid, x_test, y_valid, y_test = train_test_split(x_valid2, y_valid2, test_size=0.4, random_state=0)


    train = xgb.DMatrix(data = x_train, label = y_train)
    valid = xgb.DMatrix(data = x_valid, label = y_valid)

    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    valError = pd.DataFrame([], columns = ["rate", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["rate", "error"])

    print(y_train)

    for rate in rates:
        params = {
            "learning_rate" : rate,
        }

        model = xgb.train(params, train)



        y_preds_train = model.predict(train)
        y_preds = model.predict(valid)

        #preds = [round(value) for value in y_preds]
        #preds_train = [round(value) for value in y_preds_train]

        print(y_preds)
        print(y_preds_train)
        #print(preds)
        #print(preds_train)

        error = 1 - accuracy_score(y_valid, y_preds)
        train_error = 1 - accuracy_score(y_train, y_preds_train)

        row1 = {"rate" : rate, "error" : error}
        row3= {"rate" : rate, "error" : train_error}

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