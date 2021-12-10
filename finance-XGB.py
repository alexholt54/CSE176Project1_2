# Use finance dataset
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

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

    


if __name__ == "__main__":
    main()