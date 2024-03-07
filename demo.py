# from img_to_csv import process_image
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from train import create_model
import numpy as np
import cv2

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def process_image(image_path):
    image = cv2.imread(image_path)
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image.
    results = pose.process(image)
    
    return results.pose_landmarks


# take in an image
# convert to pose list
# load model
# give model the list
# get prediction

key = {0:'other', 1:'rukuh', 2:'sujud', 3:'wukuf'}
landmarks = process_image('./data/eval/other.png')
row_data = []
for landmark in landmarks.landmark:
    # Append x, y, z coordinates and visibility score of each landmark to the list.
    row_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

# print(row_data)
    
model = create_model()
model.load_weights('./weights.best.hdf5')
prediction = np.argmax(model.predict([row_data]))
print(key[prediction])
