from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import numpy as np
import cv2


from train import create_model


app = Flask(__name__)

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    return results.pose_landmarks

def predict_pose(image_path):
    landmarks = process_image(image_path)
    if landmarks is None:
        return "No pose detected"
    
    row_data = []
    for landmark in landmarks.landmark:
        row_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    # Assuming create_model() is defined in train.py and it's accessible
    model = create_model()
    model.load_weights('./weights.best.hdf5')
    prediction = np.argmax(model.predict(np.array([row_data]), verbose=0))
    key = {0: 'other', 1: 'rukuh', 2: 'sujud', 3: 'wukuf'}
    return key[prediction]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'picture' not in request.files:
        return 'No file part'
    file = request.files['picture']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join('./data/eval/', filename)
        file.save(save_path)
        
        # Process the saved image and get a prediction
        prediction = predict_pose(save_path)
        
        return jsonify({"message": "File uploaded and processed", "prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

