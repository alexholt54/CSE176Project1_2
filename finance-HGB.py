# Use finance dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import glob
import csv

def main():
    df = pd.read_csv("datasets/allFinance.csv")

    labels = df["PRICE VAR [%]"].values
    labels = labels.flatten()

    df = df.drop(["PRICE VAR [%]", 'Unnamed: 0'], axis=1)
    column_means = df.mean()
    df = df.fillna(column_means)
    data = df.to_numpy()

    x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.4, random_state=0)

    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.4, random_state=0)

    # Testing
    model = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.5, max_depth=500)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    print(np.sqrt(mse))

    quit()

    trees = [1, 10, 100, 300, 400, 500, 600, 700, 800, 900, 1000]

    valError = pd.DataFrame([], columns = ["trees", "error"])

    for tree in trees:
        model = HistGradientBoostingRegressor(max_iter = tree)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        mse = mean_squared_error(y_valid, pred)
        row = {"trees" : tree, "error" : np.sqrt(mse)}
        
        valError = valError.append(row, ignore_index=True)

        print(tree)

    valError.plot(x = "trees", y = "error", kind = "line", color = "red",
                        title = "HGB Root Mean Squared Error With Varying Number of Trees", ylabel = "Root Mean Squared Error", xlabel = "Number of Trees")
    plt.show()

    rates = [0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]

    valError = pd.DataFrame([], columns = ["rate", "error"])

    for rate in rates:
        model = HistGradientBoostingRegressor(max_iter=350, learning_rate=rate)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        mse = mean_squared_error(y_valid, pred)
        row = {"rate" : rate, "error" : np.sqrt(mse)}
        
        valError = valError.append(row, ignore_index=True)

        print(rate)

    valError.plot(x = "rate", y = "error", kind = "line", color = "red",
                        title = "Root Mean Squared Error With Varying Learning Rates", ylabel = "Root Mean Squared Error", xlabel = "Learning Rate")
    plt.show()

    depths = [1, 10, 50, 100, 200, 500, 1000]

    valError = pd.DataFrame([], columns = ["depth", "error"])

    for depth in depths:
        model = HistGradientBoostingRegressor(max_iter = 300, learning_rate=0.5, max_depth=depth)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        mse = mean_squared_error(y_valid, pred)
        row = {"depth" : depth, "error" : np.sqrt(mse)}
        
        valError = valError.append(row, ignore_index=True)

        print(depth)

    valError.plot(x = "depth", y = "error", kind = "line", color = "red",
                        title = "Root Mean Squared Error With Varying Max Depth Values", ylabel = "Root Mean Squared Error", xlabel = "Max Depth Values")
    plt.show()

    

    


# Call this to save data file to your machine
def loadDataFile():
    path = os.getcwd()
    csv_files = glob.glob(os.path.join("datasets/finance", "*.csv"))
    mainDF = pd.DataFrame()

    for f in csv_files:
        df = pd.read_csv(f)
        #print('Location:', f)
        print('File Name:', f.split("\\")[-1])
        df = df.drop(["operatingCycle", "cashConversionCycle", "Weighted Average Shares Diluted Growth", "Sector", "Class", "Unnamed: 0"], axis=1)
        mainDF = mainDF.append(df)

    mainDF.to_csv('allFinance.csv')

def normalizeData(data):
    train_mean = data.mean(axis = 0)
    train_std = data.std(axis = 0)
    train_std[train_std == 0] = 1
    data -= train_mean
    data /= train_std
    return data

if __name__ == "__main__":
    main()