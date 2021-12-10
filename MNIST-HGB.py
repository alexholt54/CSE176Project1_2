# Use MNIST.mat and MNIST-LeNet5.mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingClassifier

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
    lenet_test_labels = lenet_test_labels[7000:10000]

    minTrees = 100
    maxTrees = 1000

    model = HistGradientBoostingClassifier(max_iter = 700, learning_rate=0.01, l2_regularization=0.01).fit(data, label)
    lenetModel = HistGradientBoostingClassifier(max_iter = 700, learning_rate=0.01, l2_regularization=0.01).fit(lenet, lenet_labels)

    print(1 - model.score(test, test_labels))
    print(1 - lenetModel.score(lenet_test, lenet_test_labels))

    quit()

    trees = list(range(minTrees, maxTrees, 100))

    valError = pd.DataFrame([], columns = ["trees", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["trees", "error"])

    for tree in trees:
        model = HistGradientBoostingClassifier(max_iter = tree).fit(data, label)
        lenetModel = HistGradientBoostingClassifier(max_iter = tree).fit(lenet, lenet_labels)

        row = {"trees" : tree, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"trees" : tree, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(tree)

    ax = valError.plot(x = "trees", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "trees", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
                        title = "Validation Error With Varying Number of Trees", ylabel = "Validation Error", xlabel = "Number of Trees")
    plt.show()

    # For MNIST:
    # data, label
    # validation, validation_labels
    # test, test_lables

    # For MNIST LeNet5
    # lenet, lenet_labels
    # lenet_validation, lenet_validation_labels
    # lenet_test, lenet_test_labels

    minLearn = 0.1
    maxLearn = 1.1

    valError = pd.DataFrame([], columns = ["eta", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["eta", "error"])

    rates = np.arange(minLearn, maxLearn, 0.1)
    for rate in rates:
        model = HistGradientBoostingClassifier(learning_rate = rate).fit(data, label)
        lenetModel = HistGradientBoostingClassifier(learning_rate = rate).fit(lenet, lenet_labels)

        row = {"eta" : rate, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"eta" : rate, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(rate)
    ax = valError.plot(x = "eta", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "eta", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
                        title = "Validation Error With Varying Learning Rates", ylabel = "Validation Error", xlabel = "Learning Rates")
    plt.show()

    minDepth = 10
    depth = 100

    valError = pd.DataFrame([], columns = ["depth", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["depth", "error"])

    depths = list(range(minDepth, depth, 10))

    for depth in depths:
        model = HistGradientBoostingClassifier(max_depth = depth).fit(data, label)
        lenetModel = HistGradientBoostingClassifier(max_depth = depth).fit(lenet, lenet_labels)

        row = {"depth" : depth, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"depth" : depth, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(depth)
    ax = valError.plot(x = "depth", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "depth", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
                        title = "Validation Error With Varying Max Depth Values", ylabel = "Validation Error", xlabel = "Max Depth Values")
    plt.show()
    
    minL2 = 0.01
    maxL2 = 0.11

    L2s = np.arange(minL2, maxL2, 0.01)

    valError = pd.DataFrame([], columns = ["l2", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["l2", "error"])

    for l2 in L2s:
        model = HistGradientBoostingClassifier(l2_regularization=l2).fit(data, label)
        lenetModel = HistGradientBoostingClassifier(l2_regularization=l2).fit(lenet, lenet_labels)

        row = {"l2" : l2, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"l2" : l2, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(l2)
    ax = valError.plot(x = "l2", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "l2", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
                        title = "Validation Error With Varying L2 Regularization", ylabel = "Validation Error", xlabel = "L2 Regularization Values")
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