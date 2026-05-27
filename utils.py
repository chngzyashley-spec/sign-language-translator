import numpy as np

def extract_landmarks(results):
    """
    Extracts x, y coordinates from MediaPipe results and flattens them.
    We normalize coordinates relative to the first landmark (wrist).
    """
    if not results.multi_hand_landmarks:
        return None
    
    # We only take the first hand detected for simplicity
    hand_landmarks = results.multi_hand_landmarks[0]
    
    landmarks = []
    # Use the wrist as the origin (0,0)
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    
    for lm in hand_landmarks.landmark:
        landmarks.append(lm.x - base_x)
        landmarks.append(lm.y - base_y)
        
    return np.array(landmarks)
