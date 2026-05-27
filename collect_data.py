import os
import warnings
# Suppress all warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
from utils import extract_landmarks

def collect_data():
    print("\n--- SgSL DATA COLLECTION ---")
    print("1. Static (Pose - Single Frame)")
    print("2. Dynamic (Action - Sequence of Frames)")
    mode = input("Select Mode (1/2): ").strip()
    
    label = input("Enter the name of the sign (e.g., Hello): ").strip()
    if not label: return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    
    if mode == '1':
        data = []
        max_count = 100
        print(f"\nRecording {max_count} static samples for '{label}'...")
        while len(data) < max_count:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                lm = extract_landmarks(results)
                if lm is not None:
                    data.append([label] + lm.tolist())
                    mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Samples: {len(data)}/{max_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collect Static", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv('data_static.csv', mode='a', index=False, header=not os.path.exists('data_static.csv'))
            print(f"DONE! Saved to data_static.csv")

    else:
        sequences = []
        sequence_length = 30
        num_sequences = 40
        
        print(f"\nRecording {num_sequences} actions for '{label}'...")
        for seq in range(num_sequences):
            current_seq = []
            # Countdown
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"GET READY: {i}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
                cv2.imshow("Collect Dynamic", frame)
                cv2.waitKey(1000)
            
            while len(current_seq) < sequence_length:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                lm = extract_landmarks(results)
                if lm is None: lm = np.zeros(42) 
                
                current_seq.append(lm)
                if results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                
                cv2.putText(frame, "RECORDING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Collect Dynamic", frame)
                cv2.waitKey(1)
            
            sequences.append({'label': label, 'data': np.array(current_seq).flatten()})
            print(f"Action {seq+1}/{num_sequences} captured.")

        if sequences:
            df = pd.DataFrame([ [s['label']] + s['data'].tolist() for s in sequences])
            df.to_csv('data_dynamic.csv', mode='a', index=False, header=not os.path.exists('data_dynamic.csv'))
            print(f"DONE! Saved to data_dynamic.csv")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
