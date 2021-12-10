# Use finance dataset
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from math import e

def main():
    # Data reading will go here...
    df = pd.read_csv("datasets/allFinance.csv")

    labels = df["PRICE VAR [%]"].values
    labels = labels.flatten()

    trainingSet = []
    validationSet = []
    testingSet = []

    #labels
    trainingLabels = []
    validationLabels = []
    testingLabels = []

    df = df.drop("PRICE VAR [%]", axis =1)
    temp = df.iloc[0:10000]
    temp = np.array(temp)

    print(np.shape(temp))

    trainingSet = temp

    temp = df.iloc[10000:16000]
    temp = np.array(temp) 
    print(np.shape(temp))

    validationSet = temp

    temp = df.iloc[16000:21000]
    temp = np.array(temp)
    print(np.shape(temp))   

    testingSet = temp

    #training labels
    trainingLabels = labels[0:10000]
    validationLabels = labels[10000:16000]
    testingLabels = labels[16000:21000]
    
    # Change parameters here
    model = xgb.XGBRegressor()
    trees = [10, 100, 300, 375, 500]

    valError = pd.DataFrame([], columns = ["trees", "error"])

    for tree in trees:
        model = xgb.XGBRegressor(n_estimators = tree)
        model.fit(trainingSet, trainingLabels)
        pred = model.predict(validationSet)
        mse = mean_squared_error(validationLabels, pred)
        mse = 1/(1+mse^(-2))
        row = {"trees" : tree, "error" : np.sqrt(mse)}
        
        valError = valError.append(row, ignore_index=True)

        print(tree)

    ax = valError.plot(x = "trees", y = "error", kind = "line", color = "red",
                        title = "Root Mean Squared Error With Varying Number of Trees", ylabel = "Root Mean Squared Error", xlabel = "Number of Trees")
    plt.show()

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