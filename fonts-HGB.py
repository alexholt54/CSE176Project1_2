# Use fonts dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("datasets/allFont.csv")

    labels = df["font"].values
    labels = labels.flatten()
    df = df.drop(['font', 'Unnamed: 0'], axis=1)
    data = df.to_numpy()

    for i in range(len(labels)):
        if labels[i] == 'GOUDY':
            labels[i] = 0
        elif labels[i] == 'NIAGARA':
            labels[i] = 1
        elif labels[i] == 'ARIAL':
            labels[i] = 2
        elif labels[i] == 'CAMBRIA':
            labels[i] = 3
        elif labels[i] == 'COMIC':
            labels[i] = 4
        elif labels[i] == 'HARLOW':
            labels[i] = 5
        elif labels[i] == 'PAPYRUS':
            labels[i] = 6
        elif labels[i] == 'ROMAN':
            labels[i] = 7
        elif labels[i] == 'TIMES':
            labels[i] = 8
        elif labels[i] == 'MODERN':
            labels[i] = 9

    x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.33, random_state=0)

    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.4, random_state=0)

    y_train = y_train.astype('int')
    y_test = y_test.astype("int")

    trees = [1, 10, 250, 500, 1000]

    valError = pd.DataFrame([], columns = ["tree", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["tree", "error"])

    for tree in trees:
        model = HistGradientBoostingClassifier(max_iter = tree)
        model.fit(x_train, y_train)

        y_preds_train = model.predict(x_train)
        y_preds = model.predict(x_valid)

        numCorrect = 0
        for i in range(len(y_train)):
            if y_train[i] == y_preds_train[i]:
                numCorrect += 1
        train_error = 1 - (numCorrect / len(y_train))
        numCorrect = 0
        for i in range(len(y_valid)):
            if y_valid[i] == y_preds[i]:
                numCorrect += 1
        error = 1 - (numCorrect / len(y_valid))

        row1 = {"tree" : tree, "error" : error}
        row3 = {"tree" : tree, "error" : train_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)

    ax = valError.plot(x = "tree", y = "error", kind = "line", color = "red", label = "Validation")
    valErrorTrain.plot(x = "tree", y = "error", kind = "line", ax = ax, color = "blue", label = "Training",
                        title = "HGB Error With Varying Number of Trees (Fonts)", ylabel = "Error", xlabel = "Number of Trees")
    plt.show()

    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    valError = pd.DataFrame([], columns = ["rate", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["rate", "error"])

    for rate in rates:
        model = HistGradientBoostingClassifier(learning_rate=rate, max_iter=400)
        model.fit(x_train, y_train)

        y_preds_train = model.predict(x_train)
        y_preds = model.predict(x_valid)

        numCorrect = 0
        for i in range(len(y_train)):
            if y_train[i] == y_preds_train[i]:
                numCorrect += 1
        train_error = 1 - (numCorrect / len(y_train))
        numCorrect = 0
        for i in range(len(y_valid)):
            if y_valid[i] == y_preds[i]:
                numCorrect += 1
        error = 1 - (numCorrect / len(y_valid))

        row1 = {"rate" : rate, "error" : error}
        row3 = {"rate" : rate, "error" : train_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)

    ax = valError.plot(x = "rate", y = "error", kind = "line", color = "red", label = "Validation")
    valErrorTrain.plot(x = "rate", y = "error", kind = "line", ax = ax, color = "blue", label = "Training",
                        title = "HGB Error With Varying Learning Rates (Fonts)", ylabel = "Error", xlabel = "Learning Rate")
    plt.show()

    depths = [1, 10, 250, 500]

    valError = pd.DataFrame([], columns = ["depths", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["depths", "error"])

    for depth in depths:
        model = HistGradientBoostingClassifier(max_iter = 400, learning_rate=0.06, max_depth = depth)
        model.fit(x_train, y_train)

        y_preds_train = model.predict(x_train)
        y_preds = model.predict(x_valid)

        numCorrect = 0
        for i in range(len(y_train)):
            if y_train[i] == y_preds_train[i]:
                numCorrect += 1
        train_error = 1 - (numCorrect / len(y_train))
        numCorrect = 0
        for i in range(len(y_valid)):
            if y_valid[i] == y_preds[i]:
                numCorrect += 1
        error = 1 - (numCorrect / len(y_valid))

        row1 = {"depths" : depth, "error" : error}
        row3 = {"depths" : depth, "error" : train_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)

    ax = valError.plot(x = "depths", y = "error", kind = "line", color = "red", label = "Validation")
    valErrorTrain.plot(x = "depths", y = "error", kind = "line", ax = ax, color = "blue", label = "Training",
                        title = "HGB Error With Varying Max Depth Values (Fonts)", ylabel = "Error", xlabel = "Max Depth Values")
    plt.show()

    # Testing part
    model = HistGradientBoostingClassifier(learning_rate=0.2, max_iter=400, max_depth=50)
    model.fit(x_train, y_train)
    print(1 - model.score(x_test, y_test))

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