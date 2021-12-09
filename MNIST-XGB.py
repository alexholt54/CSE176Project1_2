# Use MNIST.mat and MNIST-LeNet5.mat
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def main():

    # Load data
    mnist = loadmat("datasets/MNIST.mat")
    mnistLeNet5 = loadmat("datasets/MNIST-LeNet5.mat")

    # Extract training set
    data = mnist["train_fea"]
    data = normalizeData(data)
    label = mnist["train_gnd"]
    label = label.flatten()

    lenet = mnistLeNet5["train_fea"]
    lenet = normalizeData(lenet)
    lenet_labels = mnistLeNet5["train_gnd"]
    lenet_labels = lenet_labels.flatten()

    # Extract Training and Validation Set
    test = mnist["test_fea"]
    test = normalizeData(test)
    test_labels = mnist["test_gnd"]
    test_labels = test_labels.flatten()

    lenet_test = mnistLeNet5["test_fea"]
    lenet_test = normalizeData(lenet_test)
    lenet_test_labels = mnistLeNet5["test_gnd"]
    lenet_test_labels = lenet_test_labels.flatten()

    # Extract 7000 images for validation
    validation = test[0:7000]
    validation_labels = test_labels[0:7000]

    lenet_validation = lenet_test[0:7000]
    lenet_validation_labels = lenet_test_labels[0:7000]

    # Extract 3000 images for testing
    test = test[7000:10000]
    test_labels = test_labels[7000:10000]

    lenet_test = lenet_test[7000:10000]
    lenet_test_lables = lenet_test_labels[7000:10000]

    # For MNIST:
    # data, label
    # validation, validation_labels
    # test, test_lables

    # For MNIST LeNet5
    # lenet, lenet_labels
    # lenet_validation, lenet_validation_labels
    # lenet_test, lenet_test_labels

    minTrees = 100
    maxTrees = 1000

    trees = list(range(minTrees, maxTrees, 100))

    valError = pd.DataFrame([], columns = ["trees", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["trees", "error"])

    for tree in trees:
        # Change parameters here
        model = xgb.XGBClassifier(n_estimators = tree).fit(data, label)
        lenetModel = xgb.XGBClassifier(n_estimators = tree).fit(lenet, lenet_labels)

        row = {"trees" : tree, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"trees" : tree, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(tree)

    ax = valError.plot(x = "trees", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "trees", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
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