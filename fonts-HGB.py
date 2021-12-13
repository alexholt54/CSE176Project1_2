# Use fonts dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingClassifier

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
        temp = temp.head(400)
        temp = temp.drop(['font', 'Unnamed: 0'], axis=1)
        temp = np.array(temp)
        trainingSet.extend(temp)

    for font in fontNames:
        temp = df[df['font']==font]
        temp = temp.iloc[400:530]
        temp = temp.drop(['font', 'Unnamed: 0'], axis=1)
        temp = np.array(temp)
        validationSet.extend(temp)

    for font in fontNames:
        temp = df[df['font']==font]
        temp = temp.iloc[530:610]
        temp = temp.drop(['font', 'Unnamed: 0'], axis=1)
        temp = np.array(temp)
        testingSet.extend(temp)

    for i in range(153):
        temp = [i] * 400
        temp = np.array(temp)
        trainingLabels.extend(temp)

    for i in range(153):
        temp = [i] * 130
        temp = np.array(temp)
        validationLabels.extend(temp)

    for i in range(153):
        temp = [i] * 80
        temp = np.array(temp)
        testingLabels.extend(temp)

    trainingSet = np.array(trainingSet)
    trainingSet = normalizeData(trainingSet)
    #print(type(trainingSet))
    #print(np.shape(trainingSet))

    validationSet = np.array(validationSet)
    validationSet = normalizeData(validationSet)
    #print(type(validationSet))
    #print(np.shape(validationSet))

    testingSet = np.array(testingSet)
    testingSet = normalizeData(testingSet)

    minTrees = 100
    maxTrees = 1000

    trees = [10, 50, 150, 300, 500, 1000, 2000]

    valError = pd.DataFrame([], columns = ["trees", "error"])

    model = HistGradientBoostingClassifier(max_iter = 250, learning_rate=0.1)
    model.fit(trainingSet, trainingLabels)
    print(1 - model.score(validationSet, validationLabels))
    quit()

    for tree in trees:
        model = HistGradientBoostingClassifier(max_iter = tree, learning_rate=0.1)
        model.fit(trainingSet, trainingLabels)

        row = {"trees" : tree, "error" : 1 - model.score(validationSet, validationLabels)}

        valError = valError.append(row, ignore_index=True)

        print(tree)

    ax = valError.plot(x = "trees", y = "error", kind = "line", color = "red",
                        title = "Validation Error With Varying Number of Trees", ylabel = "Validation Error", xlabel = "Number of Trees")
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