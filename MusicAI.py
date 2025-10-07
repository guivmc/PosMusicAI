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

def mfccs_feature_extractor(file_name):
    audio,sample_rate=librosa.load(file_name,res_type='kaiser_fast')
    mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features =np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plot", choices=["train", "test"],
                        help="Choose which dataset to plot (train or test)", default=None)
    args=parser.parse_args()

    # Load datasets
    dataset_test=pd.read_csv("Dataset/Musicas/Metadata_Test.csv")
    dataset_train=pd.read_csv("Dataset/Musicas/Metadata_Train.csv")

    if args.plot == "train":
        plot_dataset(dataset_train)
    elif args.plot == "test":
        plot_dataset(dataset_test)

    extracted_features=[]
    for _,row in tqdm(dataset_train.iterrows()):
        file_name=os.path.join(os.path.abspath("Dataset/Musicas/Train_submission/Train_submission"),str(row["FileName"]))
        final_class_label=row["Class"]
        mfccs_extrateced=mfccs_feature_extractor(file_name)
        extracted_features.append([mfccs_extrateced,final_class_label])

    extracted_features_data=pd.DataFrame(extracted_features,columns=['features','class'])
    print(extracted_features_data.head(5))

if __name__ == "__main__":
    main()