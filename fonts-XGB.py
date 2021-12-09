# Use fonts dataset
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import glob
import csv

def main():

    # Data reading will go here...
    # 200 images from each file ~30k
    #70 for validation/testing

    
    """     myData = pd.read_csv("datasets/fonts/AGENCY.csv")
    tempdf = pd.DataFrame(myData)
    print(tempdf)
    tempdf = tempdf.drop(['orientation', 'm_top', 'm_left', 'originalH', 'originalW', 'h', 'w', 'fontVariant'], axis=1)
    print(tempdf) """ 

    path = os.getcwd()
    csv_files = glob.glob(os.path.join("datasets/fonts", "*.csv"))

    mainDF = pd.DataFrame()

    for f in csv_files:
        df = pd.read_csv(f)
        #print('Location:', f)
        print('File Name:', f.split("\\")[-1])
        df = df.drop(['orientation', 'm_top', 'm_left', 'originalH', 'originalW', 'h', 'w', 'fontVariant'], axis=1)
        #print(df)
        mainDF = mainDF.append(df)

    print("/n")
    print(mainDF)

    #mainDF.to_csv('allFont.csv')

    #train_data = pd.read_csv("datasets/fonts/AGENCY.csv")
    #test_data = pd.read_csv("")


    # Change parameters here
    # Model = xgb.XGBClassifier()



if __name__ == "__main__":
    main()