# Use MNIST.mat and MNIST-LeNet5.mat
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

def main():

    # Load data
    mnist = loadmat("datasets/MNIST.mat")
    mnistLeNet5 = loadmat("datasets/MNIST-LeNet5.mat")

    # Extract training set
    data = mnist["train_fea"]
    data = normalizeData(data)
    label = mnist["train_gnd"]
    label = label.flatten()

    # Extract LeNet5 Training set
    lenet = mnistLeNet5["train_fea"]
    lenet = normalizeData(lenet)
    lenet_labels = mnistLeNet5["train_gnd"]
    lenet_labels = lenet_labels.flatten()

    # Extract Testing set
    test = mnist["test_fea"]
    test = normalizeData(test)
    test_labels = mnist["test_gnd"]
    test_labels = test_labels.flatten()

    # Extract LeNet5 Testing set
    lenet_test = mnistLeNet5["test_fea"]
    lenet_test = normalizeData(lenet_test)
    lenet_test_labels = mnistLeNet5["test_gnd"]
    lenet_test_labels = lenet_test_labels.flatten()

    # Remap labels to range 0 to 9
    for i in range(1,11):
        label[label == i] = i - 1
        lenet_labels[lenet_labels == i] = i - 1
        test_labels[test_labels == i] = i - 1
        lenet_test_labels[lenet_test_labels == i] = i - 1

    # Split training into training and validation
    x_train, x_valid, y_train, y_valid = train_test_split(data, label, test_size=0.2, random_state=0)

    x_train_lenet, x_valid_lenet, y_train_lenet, y_valid_lenet = train_test_split(lenet, lenet_labels, test_size=0.2, random_state=0)

    # Convert data into DMatrix
    train = xgb.DMatrix(data = x_train, label = y_train)
    valid = xgb.DMatrix(data = x_valid, label = y_valid)

    train_lenet = xgb.DMatrix(data = x_train_lenet, label = y_train_lenet)
    valid_lenet = xgb.DMatrix(data = x_valid_lenet, label = y_valid_lenet)

    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    valError = pd.DataFrame([], columns = ["rate", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["rate", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["rate", "error"])
    valErrorLenetTrain = pd.DataFrame([], columns = ["rate", "error"])

    for rate in rates:
        params = {
            "learning_rate" : rate,
        }

        model = xgb.train(params, train)
        lenet_model = xgb.train(params, train_lenet)


        y_preds_train = model.predict(train)
        y_preds = model.predict(valid)

        y_lenet_preds = lenet_model.predict(valid_lenet)
        y_lenet_train_preds = lenet_model.predict(train_lenet)

        preds = [round(value) for value in y_preds]
        lenet_preds = [round(value) for value in y_lenet_preds]
        preds_train = [round(value) for value in y_preds_train]
        lenet_train_preds = [round(value) for value in y_lenet_train_preds]

        error = 1 - accuracy_score(y_valid, preds)
        train_error = 1 - accuracy_score(y_train, preds_train)
        lenet_error = 1 - accuracy_score(y_valid_lenet, lenet_preds)
        train_lenet_error = 1 - accuracy_score(y_train_lenet, lenet_train_preds)

        row1 = {"rate" : rate, "error" : error}
        row2 = {"rate" : rate, "error" : lenet_error}
        row3 = {"rate" : rate, "error" : train_error}
        row4 = {"rate" : rate, "error" : train_lenet_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorLenet = valErrorLenet.append(row2, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)
        valErrorLenetTrain = valErrorLenetTrain.append(row4, ignore_index=True)

    ax = valError.plot(x = "rate", y = "error", kind = "line", color = "red", label = "Pixel Features (Validation)")
    ax2 = valErrorTrain.plot(x = "rate", y = "error", ax = ax, kind = "line", color = "green", label = "Pixel Features (Training)")
    ax3 = valErrorLenetTrain.plot(x = "rate", y = "error", ax = ax2, kind = "line", color = "black", label = "LeNet5 Features (Training)")
    valErrorLenet.plot(x = "rate", y = "error", kind = "line", ax = ax3, color = "blue", label = "LeNet5 Features (Validation)",
                        title = "Error With Varying Learning Rates (MNIST)", ylabel = "Error", xlabel = "Learning Rate")

    plt.show()

    depths = [1, 5, 10, 20, 30, 40, 50, 100, 500, 1000]

    valError = pd.DataFrame([], columns = ["depth", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["depth", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["depth", "error"])
    valErrorLenetTrain = pd.DataFrame([], columns = ["depth", "error"])

    for depth in depths:
        params = {
            "learning_rate" : 0.3,
            "max_depth" : depth
        }

        model = xgb.train(params, train)
        lenet_model = xgb.train(params, train_lenet)


        y_preds_train = model.predict(train)
        y_preds = model.predict(valid)

        y_lenet_preds = lenet_model.predict(valid_lenet)
        y_lenet_train_preds = lenet_model.predict(train_lenet)

        preds = [round(value) for value in y_preds]
        lenet_preds = [round(value) for value in y_lenet_preds]
        preds_train = [round(value) for value in y_preds_train]
        lenet_train_preds = [round(value) for value in y_lenet_train_preds]

        error = 1 - accuracy_score(y_valid, preds)
        train_error = 1 - accuracy_score(y_train, preds_train)
        lenet_error = 1 - accuracy_score(y_valid_lenet, lenet_preds)
        train_lenet_error = 1 - accuracy_score(y_train_lenet, lenet_train_preds)

        row1 = {"depth" : depth, "error" : error}
        row2 = {"depth" : depth, "error" : lenet_error}
        row3 = {"depth" : depth, "error" : train_error}
        row4 = {"depth" : depth, "error" : train_lenet_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorLenet = valErrorLenet.append(row2, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)
        valErrorLenetTrain = valErrorLenetTrain.append(row4, ignore_index=True)

    ax = valError.plot(x = "depth", y = "error", kind = "line", color = "red", label = "Pixel Features (Validation)")
    ax2 = valErrorTrain.plot(x = "depth", y = "error", ax = ax, kind = "line", color = "green", label = "Pixel Features (Training)")
    ax3 = valErrorLenetTrain.plot(x = "depth", y = "error", ax = ax2, kind = "line", color = "black", label = "LeNet5 Features (Training)")
    valErrorLenet.plot(x = "depth", y = "error", kind = "line", ax = ax3, color = "blue", label = "LeNet5 Features (Validation)",
                        title = "Error With Varying Max Depth Values (MNIST)", ylabel = "Error", xlabel = "Max Depth Values")

    plt.show()
    
    gammas = [0, 10, 25, 50, 75, 150, 250, 500, 850, 1000, 2000]

    valError = pd.DataFrame([], columns = ["rate", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["rate", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["rate", "error"])
    valErrorLenetTrain = pd.DataFrame([], columns = ["rate", "error"])

    for gamma in gammas:
        params = {
            "min_split_loss" : gamma
        }

        model = xgb.train(params, train)
        lenet_model = xgb.train(params, train_lenet)

        y_preds_train = model.predict(train)
        y_preds = model.predict(valid)

        y_lenet_preds = lenet_model.predict(valid_lenet)
        y_lenet_train_preds = lenet_model.predict(train_lenet)

        preds = [round(value) for value in y_preds]
        lenet_preds = [round(value) for value in y_lenet_preds]
        preds_train = [round(value) for value in y_preds_train]
        lenet_train_preds = [round(value) for value in y_lenet_train_preds]

        error = 1 - accuracy_score(y_valid, preds)
        train_error = 1 - accuracy_score(y_train, preds_train)
        lenet_error = 1 - accuracy_score(y_valid_lenet, lenet_preds)
        train_lenet_error = 1 - accuracy_score(y_train_lenet, lenet_train_preds)

        row1 = {"gamma" : gamma, "error" : error}
        row2 = {"gamma" : gamma, "error" : lenet_error}
        row3 = {"gamma" : gamma, "error" : train_error}
        row4 = {"gamma" : gamma, "error" : train_lenet_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorLenet = valErrorLenet.append(row2, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)
        valErrorLenetTrain = valErrorLenetTrain.append(row4, ignore_index=True)

    ax = valError.plot(x = "gamma", y = "error", kind = "line", color = "red", label = "Pixel Features (Validation)")
    ax2 = valErrorTrain.plot(x = "gamma", y = "error", ax = ax, kind = "line", color = "green", label = "Pixel Features (Training)")
    ax3 = valErrorLenetTrain.plot(x = "gamma", y = "error", ax = ax2, kind = "line", color = "black", label = "LeNet5 Features (Training)")
    valErrorLenet.plot(x = "gamma", y = "error", kind = "line", ax = ax3, color = "blue", label = "LeNet5 Features (Validation)",
                        title = "Error With Varying Gamma Values (MNIST)", ylabel = "Error", xlabel = "Gamma Values")

    plt.show()

    trees = [1, 100, 200, 500, 1000]

    valError = pd.DataFrame([], columns = ["tree", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["tree", "error"])
    valErrorTrain = pd.DataFrame([], columns = ["tree", "error"])
    valErrorLenetTrain = pd.DataFrame([], columns = ["tree", "error"])

    for tree in trees:
        model = XGBClassifier(n_estimators = tree)
        lenet_model = XGBClassifier(n_estimators = tree)
        model.fit(x_train, y_train)
        lenet_model.fit(x_train_lenet, y_train_lenet)

        y_preds_train = model.predict(x_train)
        y_preds = model.predict(x_valid)
        y_lenet_preds = lenet_model.predict(x_valid_lenet)
        y_lenet_train_preds = lenet_model.predict(x_train_lenet)

        preds = [round(value) for value in y_preds]
        lenet_preds = [round(value) for value in y_lenet_preds]
        preds_train = [round(value) for value in y_preds_train]
        lenet_train_preds = [round(value) for value in y_lenet_train_preds]

        error = 1 - accuracy_score(y_valid, preds)
        train_error = 1 - accuracy_score(y_train, preds_train)
        lenet_error = 1 - accuracy_score(y_valid_lenet, lenet_preds)
        train_lenet_error = 1 - accuracy_score(y_train_lenet, lenet_train_preds)

        row1 = {"tree" : tree, "error" : error}
        row2 = {"tree" : tree, "error" : lenet_error}
        row3 = {"tree" : tree, "error" : train_error}
        row4 = {"tree" : tree, "error" : train_lenet_error}

        valError = valError.append(row1, ignore_index=True)
        valErrorLenet = valErrorLenet.append(row2, ignore_index=True)
        valErrorTrain = valErrorTrain.append(row3, ignore_index=True)
        valErrorLenetTrain = valErrorLenetTrain.append(row4, ignore_index=True)

    ax = valError.plot(x = "tree", y = "error", kind = "line", color = "red", label = "Pixel Features (Validation)")
    ax2 = valErrorTrain.plot(x = "tree", y = "error", ax = ax, kind = "line", color = "green", label = "Pixel Features (Training)")
    ax3 = valErrorLenetTrain.plot(x = "tree", y = "error", ax = ax2, kind = "line", color = "black", label = "LeNet5 Features (Training)")
    valErrorLenet.plot(x = "tree", y = "error", kind = "line", ax = ax3, color = "blue", label = "LeNet5 Features (Validation)",
                        title = "Error With Varying Number of Trees (MNIST)", ylabel = "Error", xlabel = "Number of Trees")

    plt.show()

    # Testing
    model = XGBClassifier(n_estimators = 600, learning_rate=0.3, max_depth=200)
    lenet_model = XGBClassifier(n_estimators = 600, learning_rate=0.3, max_depth=200)
    model.fit(x_train, y_train)
    lenet_model.fit(x_train_lenet, y_train_lenet)

    print(1 - model.score(test, test_labels))
    print(1 - lenet_model.score(lenet_test, lenet_test_labels))

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