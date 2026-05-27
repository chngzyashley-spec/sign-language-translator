import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from utils import extract_landmarks

def translate():
    print("\n--- SgSL REAL-TIME TRANSLATOR ---")
    print("1. Static (Pose)")
    print("2. Dynamic (Action)")
    mode = input("Select Mode (1/2): ").strip()

    model_path = 'model_static.h5' if mode == '1' else 'model_dynamic.h5'
    encoder_path = 'encoder_static.pkl' if mode == '1' else 'encoder_dynamic.pkl'

    try:
        model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f: le = pickle.load(f)
    except:
        print("Error: Model not found. Did you train it?")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)
    
    sequence = []
    last_prediction = "None"
    
    print("\nStarting camera... (Press 'q' to quit)")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            lm = extract_landmarks(results)
            if lm is not None:
                if mode == '1':
                    pred = model.predict(lm.reshape(1, -1), verbose=0)
                    if np.max(pred) > 0.8:
                        new_label = le.inverse_transform([np.argmax(pred)])[0]
                        if new_label != last_prediction:
                            print(f"Detected: {new_label}")
                            last_prediction = new_label
                else:
                    sequence.append(lm)
                    sequence = sequence[-30:]
                    if len(sequence) == 30:
                        pred = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
                        if np.max(pred) > 0.85:
                            new_label = le.inverse_transform([np.argmax(pred)])[0]
                            if new_label != last_prediction:
                                print(f"Detected: {new_label}")
                                last_prediction = new_label
                            sequence = [] 

        # --- ENHANCED BIG TEXT UI ---
        color = (0, 255, 0) if mode == '1' else (0, 0, 255)
        display_text = f" {last_prediction} "
        
        # Calculate text size to draw background box correctly
        font_scale = 2.5
        thickness = 5
        (w, h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw a semi-transparent background box
        cv2.rectangle(frame, (0, 0), (w + 20, h + 40), (0, 0, 0), -1)
        
        # Draw the big text
        cv2.putText(frame, display_text, (10, h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        cv2.imshow("SgSL Translator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    translate()