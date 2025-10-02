import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
import librosa
import librosa.display
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import Adam

def plot_dataset(df):
    print(df['Class'].value_counts())
    plt.figure(figsize=(22,10))
    sns.countplot(x=df['Class'])
    plt.xticks(rotation=0)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", choices=["train", "test"],
                        help="Choose which dataset to plot (train or test)", default=None)
    args = parser.parse_args()

    # Load datasets
    dataset_test = pd.read_csv("Dataset/Musicas/Metadata_Test.csv")
    dataset_train = pd.read_csv("Dataset/Musicas/Metadata_Train.csv")

    if args.plot == "train":
        plot_dataset(dataset_train)
    elif args.plot == "test":
        plot_dataset(dataset_test)

if __name__ == "__main__":
    main()