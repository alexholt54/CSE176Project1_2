# Use finance dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingRegressor
import os
import glob
import csv

def main():

    # Data reading will go here...

    loadDataFile()
    df = pd.read_csv("datasets/allFinance.csv")

    print(np.shape(df))

    df = df.fillna(0)

    print(np.shape(df))

    labels = df["PRICE VAR [%]"].values
    labels = labels.flatten()

    print(np.shape(labels))

    # Change parameters here
    model = HistGradientBoostingRegressor()


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

if __name__ == "__main__":
    main()