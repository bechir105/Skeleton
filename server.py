from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import numpy as np
import cv2
from pymongo import MongoClient
from datetime import datetime
import bson

from train import create_model

app = Flask(__name__)

# MongoDB setup
client = MongoClient('mongodb+srv://mohamedbechirkefi:MGtxYZUPUchNkmtl@skeleton.mawhsat.mongodb.net/?retryWrites=true&w=majority&appName=Skeleton')
db = client['salatImages']
collection = db['skeleton']
# Set up a TTL index for automatic deletion after 604800 seconds (7 days)
collection.create_index("created_at", expireAfterSeconds=604800)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def process_image(image_bytes):
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    return results.pose_landmarks

def predict_pose(image_bytes):
    landmarks = process_image(image_bytes)
    if landmarks is None:
        return "No pose detected"
    
    row_data = []
    for landmark in landmarks.landmark:
        row_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
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
        image_bytes = file.read()  # Read file as bytes

        # Process the image bytes and get a prediction
        prediction = predict_pose(image_bytes)
        
        # Store the image and metadata in MongoDB
        file_document = {
            'filename': filename,
            'file_data': bson.binary.Binary(image_bytes),
            'created_at': datetime.utcnow(),
            'prediction': prediction
        }
        collection.insert_one(file_document)
        
        return jsonify({"message": "File uploaded and processed", "prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
