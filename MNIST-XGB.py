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

    image1s = data[label == 1]
    image2s = data[label == 2]
    image3s = data[label == 3]
    image4s = data[label == 4]
    image5s = data[label == 5]
    image6s = data[label == 6]
    image7s = data[label == 7]
    image8s = data[label == 8]
    image9s = data[label == 9]
    image10s = data[label == 10]

    numImages = 3000

    train = np.concatenate((image1s[0:numImages], image2s[0:numImages], image3s[0:numImages], image4s[0:numImages], image5s[0:numImages], image6s[0:numImages], image7s[0:numImages],
                            image8s[0:numImages], image9s[0:numImages], image10s[0:numImages]))

    train_label = np.concatenate(([0] * numImages, [1] * numImages, [2] * numImages, [3] * numImages, [4] * numImages, [5] * numImages, [6] * numImages, [7] * numImages, [8] * numImages,
                                    [9] * numImages))

    lenet = mnistLeNet5["train_fea"]
    lenet = normalizeData(lenet)
    lenet_labels = mnistLeNet5["train_gnd"]
    lenet_labels = lenet_labels.flatten()

    image1s = lenet[lenet_labels == 1]
    image2s = lenet[lenet_labels == 2]
    image3s = lenet[lenet_labels == 3]
    image4s = lenet[lenet_labels == 4]
    image5s = lenet[lenet_labels == 5]
    image6s = lenet[lenet_labels == 6]
    image7s = lenet[lenet_labels == 7]
    image8s = lenet[lenet_labels == 8]
    image9s = lenet[lenet_labels == 9]
    image10s = lenet[lenet_labels == 10]

    lenet = np.concatenate((image1s[0:numImages], image2s[0:numImages], image3s[0:numImages], image4s[0:numImages], image5s[0:numImages], image6s[0:numImages], image7s[0:numImages],
                            image8s[0:numImages], image9s[0:numImages], image10s[0:numImages]))

    lenet_labels = np.concatenate(([0] * numImages, [1] * numImages, [2] * numImages, [3] * numImages, [4] * numImages, [5] * numImages, [6] * numImages, [7] * numImages, [8] * numImages,
                                    [9] * numImages))

    # Extract Training and Validation Set
    test = mnist["test_fea"]
    test = normalizeData(test)
    test_labels = mnist["test_gnd"]
    test_labels = test_labels.flatten()

    for i in range(1, 11):
        test_labels[test_labels == i] = i - 1

    lenet_test = mnistLeNet5["test_fea"]
    lenet_test = normalizeData(lenet_test)
    lenet_test_labels = mnistLeNet5["test_gnd"]
    lenet_test_labels = lenet_test_labels.flatten()

    for i in range(1, 11):
        lenet_test_labels[lenet_test_labels == i] = i - 1

    # Extract 6000 images for validation
    validation = test[0:6000]
    validation_labels = test_labels[0:6000]

    lenet_validation = lenet_test[0:6000]
    lenet_validation_labels = lenet_test_labels[0:6000]

    # Extract 4000 images for testing
    test = test[6000:10000]
    test_labels = test_labels[6000:10000]

    lenet_test = lenet_test[6000:10000]
    lenet_test_labels = lenet_test_labels[6000:10000]

    # For MNIST:
    # train, train_label
    # validation, validation_labels
    # test, test_lables

    # For MNIST LeNet5
    # lenet, lenet_labels
    # lenet_validation, lenet_validation_labels
    # lenet_test, lenet_test_labels

    model = xgb.XGBClassifier(n_estimators = 600, learning_rate = 0.01, use_label_encoder=False)
    model.fit(lenet, lenet_labels)

    print(1 - model.score(lenet_test, lenet_test_labels))
    quit()

    minTrees = 100
    maxTrees = 1000

    trees = list(range(minTrees, maxTrees, 100))

    valError = pd.DataFrame([], columns = ["trees", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["trees", "error"])

    for tree in trees:
        # Change parameters here
        model = xgb.XGBClassifier(n_estimators = tree, use_label_encoder=False)
        model.fit(train, train_label)
        lenetModel = xgb.XGBClassifier(n_estimators = tree, use_label_encoder=False)
        lenetModel.fit(lenet, lenet_labels)

        row = {"trees" : tree, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"trees" : tree, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(tree)

    ax = valError.plot(x = "trees", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "trees", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
                        title = "Validation Error With Varying Number of Trees", ylabel = "Validation Error", xlabel = "Number of Trees")

    minLearn = 0.1
    maxLearn = 1

    valError = pd.DataFrame([], columns = ["eta", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["eta", "error"])

    # For MNIST:
    # data, label
    # validation, validation_labels
    # test, test_lables

    # For MNIST LeNet5
    # lenet, lenet_labels
    # lenet_validation, lenet_validation_labels
    # lenet_test, lenet_test_labels

    rates = np.arange(0.1, 1.1, 0.1)
    for rate in rates:
        model = xgb.XGBClassifier(learning_rate = rate).fit(data, label)
        lenetModel = xgb.XGBClassifier(learning_rate = rate).fit(lenet, lenet_labels)

        row = {"eta" : rate, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"eta" : rate, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(rate)
    ax = valError.plot(x = "eta", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "eta", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
                        title = "Validation Error With Varying Learning Rates", ylabel = "Validation Error", xlabel = "Learning Rates")
    plt.show()

    minDepth = 0
    depth = 1000

    valError = pd.DataFrame([], columns = ["depth", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["depth", "error"])

    depths = list(range(minDepth, depth, 100))

    for depth in depths:
        model = xgb.XGBClassifier(max_depth = depth).fit(data, label)
        lenetModel = xgb.XGBClassifier(max_depth = depth).fit(lenet, lenet_labels)

        row = {"depth" : depth, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"depth" : depth, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(depth)
    ax = valError.plot(x = "depth", y = "error", kind = "line", color = "red", label = "Pixel Features")
    valErrorLenet.plot(x = "depth", y = "error", kind = "line", ax = ax, color = "blue", label = "LeNet5 Features",
                        title = "Validation Error With Varying Max Depth Values", ylabel = "Validation Error", xlabel = "Max Depth Values")
    plt.show()
    
    minL2 = 0
    maxL2 = 0.1

    L2s = np.arange(0.0, 1.1, 0.1)

    valError = pd.DataFrame([], columns = ["l2", "error"])
    valErrorLenet = pd.DataFrame([], columns = ["l2", "error"])

    for l2 in L2s:
        model = xgb.XGBClassifier(l2_regularization=l2).fit(data, label)
        lenetModel = xgb.XGBClassifier(l2_regularization=l2).fit(lenet, lenet_labels)

        row = {"l2" : l2, "error" : 1 - model.score(validation, validation_labels)}
        lenet_row = {"l2" : l2, "error" : 1 - lenetModel.score(lenet_validation, lenet_validation_labels)}

        valError = valError.append(row, ignore_index=True)
        valErrorLenet = valErrorLenet.append(lenet_row, ignore_index=True)

        print(depth)
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