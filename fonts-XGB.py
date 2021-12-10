# Use fonts dataset
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
import os
import glob
import csv

def main():

    # Data reading will go here...
    # 200 images from each file ~30k
    #70 for validation/testing

    df = pd.read_csv("datasets/allFont.csv")

    print(np.shape(df))

    labels = df["font"].values
    labels = labels.flatten()

    print(labels)

    le = preprocessing.LabelEncoder()
    labels = le.transform(labels)

    print(labels)
    
    # Change parameters here
    # Model = xgb.XGBClassifier()

# Call this to save data file to your machine
def loadDataFile():
    path = os.getcwd()
    csv_files = glob.glob(os.path.join("datasets/fonts", "*.csv"))
    mainDF = pd.DataFrame()

    for f in csv_files:
        df = pd.read_csv(f)
        #print('Location:', f)
        print('File Name:', f.split("\\")[-1])
        df = df.drop(['orientation', 'm_top', 'm_left', 'originalH', 'originalW', 'h', 'w', 'fontVariant'], axis=1)
        mainDF = mainDF.append(df)

    mainDF.to_csv('allFont.csv')



if __name__ == "__main__":
    main()