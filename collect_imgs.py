import os
import cv2
import string
from tqdm import tqdm

# Constants
DATA_DIR = './data'
STATIC_LETTERS = [
    letter for letter in string.ascii_uppercase if letter not in ['J', 'Z', 'Q', 'X']
]
DATASET_SIZE = 100

# Ensure the base directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Create folders numbered 0 to (n-1) for static letters
for idx, letter in enumerate(STATIC_LETTERS):
    folder_name = str(idx)  # Numbers instead of letters for indexes
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Open video capture
cap = cv2.VideoCapture(0)  # Use webcam (adjust index if needed)

print("Static Sign Language Data Collection Tool")
print("Static letters: " + ", ".join(STATIC_LETTERS))
print("Press 'q' to start collecting data for a specific letter.")
print("Press 'ESC' to exit the tool at any time.\n")

while True:
    print("\nWhich letter would you like to collect data for?")
    for idx, letter in enumerate(STATIC_LETTERS):
        print(f"{idx}: {letter}")
    
    try:
        selected_index = int(input(f"Enter the number (0-{len(STATIC_LETTERS)-1}): "))
        if selected_index < 0 or selected_index >= len(STATIC_LETTERS):
            raise ValueError("Index out of range.")
    except ValueError as e:
        print(f"Invalid input. Please enter a number between 0 and {len(STATIC_LETTERS)-1}.")
        continue

    selected_letter = STATIC_LETTERS[selected_index]
    print(f"\nCollecting data for letter: {selected_letter} (Folder: {selected_index})")
    print("Press 'q' to begin. Get ready to display the sign.")

    # Wait for user to confirm readiness
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f"Get ready for {selected_letter}. Press 'q' to start.", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Collect dataset
    counter = 0
    progress_bar = tqdm(total=DATASET_SIZE, desc=f"Collecting {selected_letter}", unit="image")

    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access camera.")
            break

        # Display frame with instructions
        cv2.putText(frame, f"Collecting {selected_letter}: {counter}/{DATASET_SIZE}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Data Collection', frame)

        # Save frame
        file_path = os.path.join(DATA_DIR, str(selected_index), f"{counter}.jpg")
        cv2.imwrite(file_path, frame)
        counter += 1
        progress_bar.update(1)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            print("Exiting...")
            break

    progress_bar.close()

    print(f"Data collection for letter {selected_letter} completed.")
    print("Press 'q' to collect data for another letter, or 'ESC' to quit.\n")

    # Check if the user wants to quit
    ret, frame = cap.read()
    cv2.putText(frame, "Press 'q' to collect for another letter or ESC to exit.", 
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Data Collection', frame)
    key = cv2.waitKey(0)
    if key & 0xFF == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
