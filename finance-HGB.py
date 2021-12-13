# Use finance dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import glob
import csv

def main():
    df = pd.read_csv("datasets/allFinance.csv")

    labels = df["PRICE VAR [%]"].values
    labels = labels.flatten()

    df = df.drop(["PRICE VAR [%]", "Unnamed: 0"], axis =1)
    df = df.to_numpy()


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