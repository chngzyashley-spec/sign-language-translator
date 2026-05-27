import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def train():
    print("\n--- SgSL MODEL TRAINING ---")
    print("1. Static (Pose)")
    print("2. Dynamic (Action)")
    mode = input("Select Model to Train (1/2): ").strip()
    
    filename = 'data_static.csv' if mode == '1' else 'data_dynamic.csv'
    model_name = 'model_static.h5' if mode == '1' else 'model_dynamic.h5'
    encoder_name = 'encoder_static.pkl' if mode == '1' else 'encoder_dynamic.pkl'

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(filename)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    with open(encoder_name, 'wb') as f: pickle.dump(le, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

    if mode == '1':
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(42,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(le.classes_), activation='softmax')
        ])
    else:
        X_train = X_train.reshape(-1, 30, 42)
        X_test = X_test.reshape(-1, 30, 42)
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 42)),
            layers.LSTM(128, return_sequences=False, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(le.classes_), activation='softmax')
        ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Training started...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    model.save(model_name)
    print(f"\nSUCCESS! Model saved as {model_name}")

if __name__ == "__main__":
    train()
