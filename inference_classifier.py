import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load model and labels
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  
    min_detection_confidence=0.3,
    model_complexity=1  
)

labels_dict = {
    0: 'A', 
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E', 
    5: 'F', 
    6: 'G', 
    7: 'H', 
    8: 'I', 
    9: 'K', 
    10: 'L', 
    11: 'M', 
    12: 'N', 
    13: 'O', 
    14: 'P', 
    15: 'R', 
    16: 'S', 
    17: 'T', 
    18: 'U', 
    19: 'V', 
    20: 'W', 
    21: 'Y'
}

while True:
    x_ = []
    y_ = []
    data_aux = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
             mp_drawing.draw_landmarks(
                  frame,
                  hand_landmarks, 
                  mp_hands.HAND_CONNECTIONS, 
                  mp_drawing_styles.get_default_hand_landmarks_style(), 
                  mp_drawing_styles.get_default_hand_connections_style())
             
        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

        x1 = int(min(x_) * W)
        x2 = int(max(x_) * W)

        y1 = int(min(y_) * H)
        y2 = int(max(y_) * H)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_char = labels_dict[int(prediction[0])]

        # Adjust rectangle positioning
        rect_height = 40
        rect_y1 = y1 - rect_height - 10  # Rectangle above the bounding box
        rect_y2 = y1 - 10

        # Ensure rectangle doesn't go out of bounds
        if rect_y1 < 0:
            rect_y1 = 0
            rect_y2 = rect_height

        # Draw the rectangle and text
        cv2.rectangle(frame, (x1, rect_y1), (x2, rect_y2), (0, 0, 0), -1)  # Filled rectangle
        text_size = cv2.getTextSize(predicted_char, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = rect_y2 - 10

        cv2.putText(frame, predicted_char, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Sign Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
