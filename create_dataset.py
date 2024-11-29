import mediapipe as mp
import cv2
import os
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,  
    min_detection_confidence=0.5,  
    model_complexity=1  
)

DATA_DIR = './data'

def process_image(img_path, label):
    """
    Process a single image and extract hand landmarks.
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)
        return data_aux, label
    return None, None

def load_data():
    """
    Load and process all images from DATA_DIR.
    """
    data = []
    labels = []

    # ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = []
        for dirpath, _, filenames in os.walk(DATA_DIR):
            label = os.path.basename(dirpath)
            for img_file in filenames:
                img_path = os.path.join(dirpath, img_file)
                futures.append(executor.submit(process_image, img_path, label))
        
        # collecting results
        for future in futures:
            result, label = future.result()
            if result:
                data.append(result)
                labels.append(label)

    return np.array(data), np.array(labels)

# processing and saving data
data, labels = load_data()
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data processed and saved. Total samples: {len(data)}")
