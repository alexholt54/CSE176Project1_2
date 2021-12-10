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

def main():

    # Data reading will go here...
    # 200 images from each file ~30k
    #70 for validation/testing

    df = pd.read_csv("datasets/allFont.csv")

    #print(np.shape(df))

    labels = df["font"].values
    labels = labels.flatten()

    #data
    trainingSet = []
    validationSet = []
    testingSet = []

    #labels
    trainingLabels = []
    validationLabels = []
    testingLabels = []

    fontNames = []

    for label in labels:
        if label not in fontNames:
            fontNames.append(label)

    for font in fontNames:
        temp = df[df['font']==font]
        temp = temp.head(200)
        temp = np.array(temp)
        trainingSet.extend(temp)

    for font in fontNames:
        temp = df[df['font']==font]
        temp = temp.iloc[201:266]
        temp = np.array(temp)
        validationSet.extend(temp)

    for font in fontNames:
        temp = df[df['font']==font]
        temp = temp.iloc[267:307]
        temp = np.array(temp)
        testingSet.extend(temp)

    for i in range(153):
        temp = [0] * 200
        temp = np.array(temp)
        trainingLabels.extend(temp)

    for i in range(153):
        temp = [0] * 65
        temp = np.array(temp)
        validationLabels.extend(temp)

    for i in range(153):
        temp = [0] * 40
        temp = np.array(temp)
        testingLabels.extend(temp)

    print(np.shape(trainingLabels))
    print(np.shape(validationLabels))
    print(np.shape(testingLabels))

    trainingSet = np.array(trainingSet)
    #print(type(trainingSet))
    #print(np.shape(trainingSet))

    validationSet = np.array(validationSet)
    #print(type(validationSet))
    #print(np.shape(validationSet))

    testingSet = np.array(testingSet)
    #print(type(testingSet))
    #print(np.shape(testingSet))

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)

    df.drop(['font'], axis=1)

    #print(np.shape(df))

    #print(labels)

    # Change parameters here
    Model = xgb.XGBClassifier()

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