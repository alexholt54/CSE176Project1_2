# Use MNIST.mat and MNIST-LeNet5.mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingClassifier

def main():

    # Data reading will go here...

    # Change parameters here
    model = HistGradientBoostingClassifier()

if __name__ == "__main__":
    main()