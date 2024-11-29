import os
import cv2
import string
from tqdm import tqdm

DATA_DIR = './data'
STATIC_LETTERS = [
    letter for letter in string.ascii_uppercase if letter not in ['J', 'Z', 'Q', 'X']
]
DATASET_SIZE = 300  

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# create folders numbered 0 to (n-1) for static letters
for idx, letter in enumerate(STATIC_LETTERS):
    folder_name = str(idx)  # numbers instead of letters for indexes
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Open video capture
cap = cv2.VideoCapture(0)  # use webcam (adjust index if needed)

print("Static Sign Language Data Collection Tool")
print("Static letters: " + ", ".join(STATIC_LETTERS))
print("Press 'q' when ready to start collecting data for each letter in sequence.")
print("Press 'ESC' to exit the tool at any time.\n")

for idx, selected_letter in enumerate(STATIC_LETTERS):
    print(f"\nCollecting data for letter: {selected_letter} (Folder: {idx})")
    print("Press 'q' to begin. Get ready to display the sign.")

    # wait for user to confirm readiness
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f"Get ready for {selected_letter}. Press 'q' to start.", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # collect dataset
    counter = 0
    progress_bar = tqdm(total=DATASET_SIZE, desc=f"Collecting {selected_letter}", unit="image")

    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access camera.")
            break

        cv2.putText(frame, f"Collecting {selected_letter}: {counter}/{DATASET_SIZE}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Data Collection', frame)

        # save frame
        file_path = os.path.join(DATA_DIR, str(idx), f"{counter}.jpg")
        cv2.imwrite(file_path, frame)
        counter += 1
        progress_bar.update(1)

        # exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            print("Exiting...")
            break

    progress_bar.close()

    print(f"Data collection for letter {selected_letter} completed.")
    print("Get ready for the next letter.\n")
    
    # pause between letters
    for i in range(3, 0, -1):
        print(f"Starting next letter in {i} seconds...")
        ret, frame = cap.read()
        cv2.putText(frame, f"Starting next letter in {i} seconds...", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Data Collection', frame)
        cv2.waitKey(1000)

    # check if ESC key is pressed to exit
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
