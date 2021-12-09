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
    label = mnist["train_gnd"]

    data_lenet = mnistLeNet5["train_fea"]
    label_lenet = mnistLeNet5["train_gnd"]

    # Normalizing data
    data = (data / 255)
    train_mean = data.mean(axis = 0)
    train_std = data.std(axis = 0)
    train_std[train_std == 0] = 1
    data -= train_mean
    data /= train_std
    label = label.flatten()

    data_lenet = (data_lenet / 255)
    lenet_mean = data_lenet.mean(axis = 0)
    lenet_std = data_lenet.std(axis = 0)
    lenet_std[lenet_std == 0] = 1
    data_lenet -= lenet_mean
    data_lenet /= lenet_std
    label_lenet = label_lenet.flatten()

    # Split data into training and validation
    num_training = 50000 # number of images to use in the training set

    # Training set
    data_train = data[:num_training]
    labels_train = label[:num_training]

    # Training set for lenet
    lenet_train = data_lenet[:num_training]
    lenet_labels = label_lenet[:num_training]

    # Validation set will be 60000 - num_training
    data_validation = data[num_training:]
    labels_validation = label[num_training:]

    # Validation set for lenet
    lenet_validation = data_lenet[num_training:]
    lenet_labels_validation = label_lenet[num_training:]

    # Extract test set to use at the end
    data_test = mnist["test_fea"]
    label_test = mnist["test_gnd"]

    lenet_test = mnistLeNet5["test_fea"]
    lenet_label_test = mnistLeNet5["test_gnd"]

    # Normalizing testing data
    data_test = (data_test / 255)
    test_mean = data_test.mean(axis = 0)
    test_std = data_test.std(axis = 0)
    test_std[test_std == 0] = 1
    data_test -= test_mean
    data_test /= test_std
    label_test = label_test.flatten()

    lenet_test = (lenet_test / 255)
    lenet_test_mean = lenet_test.mean(axis = 0)
    lenet_test_std = lenet_test.std(axis = 0)
    lenet_test_std[lenet_test_std == 0] = 1
    lenet_test -= lenet_test_mean
    lenet_test /= lenet_test_std
    lenet_label_test = lenet_label_test.flatten()

    # For MNIST:
    # data_train, labels_train
    # data_validation, labels_validation
    # data_test, label_test

    # For MNIST LeNet5
    # lenet_train, lenet_labels
    # lenet_validation, lenet_labels_validation
    # lenet_test, lenet_label_test

    minTrees = 1
    maxTrees = 500

    trees = list(range(minTrees, maxTrees))

    valError = pd.DataFrame([], columns = ["trees", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["trees", "error"])

    for tree in trees:
        # Change parameters here
        model = HistGradientBoostingClassifier(max_iter = tree).fit(data_train, labels_train)
        lenet = HistGradientBoostingClassifier(max_iter = tree).fit(lenet_train, lenet_labels)

        row = {"trees" : tree, "error" : 1 - model.score(data_validation, labels_validation)}
        lenet_row = {"trees" : tree, "error" : 1 - lenet.score(lenet_validation, lenet_labels_validation)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valError.append(lenet_row, ignore_index=True)

        print(tree)

    ax = valError.plot(x = "trees", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "trees", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
                        title = "Validation Error With Varying Number of Trees", ylabel = "Validation Error", xlabel = "Number of Trees")
    plt.show()


if __name__ == "__main__":
    main()