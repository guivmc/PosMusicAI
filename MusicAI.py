import ast
import os
from datetime import datetime
import random
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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def plot_dataset(df):
    print(df['Class'].value_counts())
    plt.figure(figsize=(22,10))
    sns.countplot(x=df['Class'])
    plt.xticks(rotation=0)
    plt.show()

def mfccs_feature_extractor(file_name):
    """
       Computes the mean across all time frames for each MFCC coefficient, resulting in a 1D vector.
    """
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

def parse_feature_string(s):
    # remove brackets if present, split by spaces, convert to float
    s = s.strip("[]")
    return np.array([float(x) for x in s.split()], dtype=np.float32)

def create_model(labels):
    model=Sequential()
    model.add(Dense(128,input_shape=(40,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(labels))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    model.summary()
    return model

def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "accuracy")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_accuracy")
    plt.title("train_accuracy vs val_accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plot", choices=["train", "test"],
                        help="Choose which dataset to plot (train or test)", default=None)
    parser.add_argument("--extract", help="Extract features from train data set", action="store_true")
    parser.add_argument("--createModel", help="Create a new model", action="store_true")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=5000)
    parser.add_argument("--batchSize", type=int, help="Batch size", default=64)
    args=parser.parse_args()

    # Load datasets
    dataset_test=pd.read_csv("Dataset/Musicas/Metadata_Test.csv")
    dataset_train=pd.read_csv("Dataset/Musicas/Metadata_Train.csv")

    if args.plot == "train":
        plot_dataset(dataset_train)
    elif args.plot == "test":
        plot_dataset(dataset_test)

    # Extract features
    if args.extract:
        dataset_features=extract_features(dataset_train)
        dataset_features.to_csv('extracted_features.csv', sep=',', encoding='utf-8', index=False, header=True)
    else:
        dataset_features=pd.read_csv("extracted_features.csv")
        dataset_features["features"] = dataset_features["features"].apply(parse_feature_string)

    # Get Labels
    LE=LabelEncoder()
    x_axis=np.array(dataset_features["features"].tolist(), dtype=np.float32)
    classes=dataset_features["class"].tolist()
    y_axis=np.array(classes)
    y_axis=to_categorical(LE.fit_transform(y_axis))

    # Train new model
    if args.createModel:
        x_train,x_test,y_train,y_test=train_test_split(x_axis, y_axis,test_size=0.2,random_state=42)

        num_labels=y_axis.shape[1]
        created_model=create_model(num_labels)

        num_epochs=args.epochs
        num_batch_size=args.batchSize

        checkpointer=ModelCheckpoint(filepath="saved_models/gladiador_audio_classification.keras",verbose=1,save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=20,  restore_best_weights=True)
        start=datetime.now()
        history = created_model.fit(
            x_train, y_train,
            batch_size=num_batch_size,
            epochs=num_epochs,
            validation_data=(x_test, y_test),
            callbacks=[checkpointer, early_stop]
        )

        duration=datetime.now()-start
        print("Training Completed in time: ",duration)
        plot_accuracy_loss(history)
        test_accuracy=created_model.evaluate(x_test,y_test,batch_size=128,verbose=0)
        print(test_accuracy[1])


    # Test model
    model=load_model("saved_models/gladiador_audio_classification.keras")
    folder_path="Dataset/Musicas/Test_submission/Test_submission"
    y_true=[]
    y_pred=[]

    for index, row in dataset_test.iterrows():
        print("File to test:", row["FileName"])
        print("Expected class:", row["Class"])

        y_true.append(row["Class"])

        # Extract features
        audio,sample_rate=librosa.load(os.path.join(folder_path, row["FileName"]),res_type='kaiser_fast')
        mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
        mfccs_scaled_features =np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

        # Predict using model
        predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=-1)
        prediction_class=LE.inverse_transform(predicted_label)
        y_pred.append(prediction_class)

    # Create confusion matrix
    cm=confusion_matrix(y_true, y_pred, labels=LE.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LE.classes_, yticklabels=LE.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=LE.classes_))

if __name__ == "__main__":
    main()