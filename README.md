
# Hand Sign Language Detection

This project detects hand signs for the alphabet using a webcam and machine learning. It uses MediaPipe for hand landmark detection and a trained Random Forest model for classification.

## How It Works
1. **Data Collection**: Use a webcam to capture images for different hand signs.
2. **Dataset Creation**: Process the images to extract hand landmarks.
3. **Model Training**: Train a model to classify hand signs.
4. **Real-Time Detection**: Use the trained model to predict hand signs in real-time with your webcam.

---

## Usage
1. **Collect Images**:  
   Run the script to collect hand sign images:
   ```bash
   python collect_imgs.py
   ```
   - Use your webcam to display each sign.
   - Saves images to the `data/` directory.

2. **Create Dataset**:  
   Process the images to create a dataset:
   ```bash
   python create_dataset.py
   ```

3. **Train the Model**:  
   Train the Random Forest model:
   ```bash
   python train_classifier.py
   ```

4. **Real-Time Detection**:  
   Use your webcam to predict hand signs:
   ```bash
   python inference_classifier.py
   ```

---

## Files
- **`collect_imgs.py`**: Captures hand sign images.
- **`create_dataset.py`**: Processes images to extract landmarks.
- **`train_classifier.py`**: Trains the Random Forest model.
- **`inference_classifier.py`**: Runs real-time sign detection using the trained model.

---
