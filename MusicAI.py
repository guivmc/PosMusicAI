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
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

def extract_features(dataset):
    extracted_features=[]
    for _,row in tqdm(dataset.iterrows()):
        file_name=os.path.join(os.path.abspath("Dataset/Musicas/Train_submission/Train_submission"),str(row["FileName"]))
        final_class_label=row["Class"]
        mfccs_extrateced=mfccs_feature_extractor(file_name)
        extracted_features.append([mfccs_extrateced,final_class_label])
    
    extracted_features_dataframe=pd.DataFrame(extracted_features,columns=['features','class'])
    return extracted_features_dataframe

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

    dataset_features=extract_features(dataset_train)
    print(dataset_features.head(5))

    x_axis=np.array(dataset_features['features'].tolist())
    y_axis=np.array(dataset_features["class"].tolist())

    LE=LabelEncoder()
    y_axis=to_categorical(LE.fit_transform(y_axis))

    x_train,x_test,y_train,y_test=train_test_split(x_axis, y_axis,test_size=0.2,random_state=42)

    print("x_train:",x_train.shape)
    print("x_test:",x_test.shape)
    print("y_train:",y_train.shape)
    print("y_test:",y_test.shape)

    num_labels=y_axis.shape[1]
    print(num_labels)

if __name__ == "__main__":
    main()