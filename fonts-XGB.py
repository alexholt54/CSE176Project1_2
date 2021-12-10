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
        temp = temp.drop(['font', 'Unnamed: 0'], axis=1)
        temp = np.array(temp)
        trainingSet.extend(temp)

    for font in fontNames:
        temp = df[df['font']==font]
        temp = temp.iloc[201:266]
        temp = temp.drop(['font', 'Unnamed: 0'], axis=1)
        temp = np.array(temp)
        validationSet.extend(temp)

    for font in fontNames:
        temp = df[df['font']==font]
        temp = temp.iloc[267:307]
        temp = temp.drop(['font', 'Unnamed: 0'], axis=1)
        temp = np.array(temp)
        testingSet.extend(temp)

    for i in range(153):
        temp = [i] * 200
        temp = np.array(temp)
        trainingLabels.extend(temp)

    for i in range(153):
        temp = [i] * 65
        temp = np.array(temp)
        validationLabels.extend(temp)

    for i in range(153):
        temp = [i] * 40
        temp = np.array(temp)
        testingLabels.extend(temp)

    trainingSet = np.array(trainingSet)
    #print(type(trainingSet))
    #print(np.shape(trainingSet))

    validationSet = np.array(validationSet)
    #print(type(validationSet))
    #print(np.shape(validationSet))

    testingSet = np.array(testingSet)
    #print(type(testingSet))
    #print(np.shape(testingSet))

    #print(np.shape(df))

    #print(labels)

    # Change parameters here
        
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
        df = df.drop(['orientation', 'm_top', 'm_left', 'originalH', 'originalW', 'h', 'w', 'fontVariant'], axis=1)
        mainDF = mainDF.append(df)

    mainDF.to_csv('allFont.csv')



if __name__ == "__main__":
    main()