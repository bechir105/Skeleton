import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Function to process an image and detect poses.
def process_image(image_path, aug=False):
    image = cv2.imread(image_path)
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if aug:
        image = cv2.flip(image, 1)
    # Process the image.
    results = pose.process(image)
    
    return results.pose_landmarks

# Function to save landmarks to a CSV file.
def save_landmarks_to_csv(landmarks, csv_file_path, class_name):
    file_exists = Path(csv_file_path).is_file()
    file_name = os.path.basename(csv_file_path)

    # Check if landmarks were detected.
    if landmarks:
        # Initialize an empty list to store landmark data.
        row_data = []
        for landmark in landmarks.landmark:
            # Append x, y, z coordinates and visibility score of each landmark to the list.
            row_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        # Create a DataFrame with a single row.
        # row_data = list(np.delete(row_data, range(2, 29)))
        row_data.append(class_name)
        row_data.insert(0, file_name)
        df = pd.DataFrame([row_data])
        # Generate column names based on landmark indices and attributes.
        columns = [f'{attribute}_{index}' for index in range(len(landmarks.landmark)) for attribute in ('x', 'y', 'z', 'visibility')]
        # columns = list(np.delete(columns, range(2, 29)))
        columns.append('class_name')
        columns.insert(0, 'file_name')
        df.columns = columns
        # Save to CSV.
        df.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)


def generate_class(class_name, csv_file_path, img_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            # print(f"Processing file: {file_path}")
            landmarks = process_image(file_path)
            save_landmarks_to_csv(landmarks, csv_file_path, class_name)

            data_augment = process_image(file_path, True)
            save_landmarks_to_csv(landmarks, csv_file_path, class_name)



# training data
print("generating training data")

# wukuf
csv_file_path = 'train_data.csv'
directory_path = './data/train/wukuf'
generate_class('wukuf', csv_file_path, directory_path)

# rukuh
csv_file_path = 'train_data.csv'
directory_path = './data/train/rukuh'
generate_class('rukuh', csv_file_path, directory_path)

# sujud
csv_file_path = 'train_data.csv'
directory_path = './data/train/sujud'
generate_class('sujud', csv_file_path, directory_path)

# other
csv_file_path = 'train_data.csv'
directory_path = './data/train/other'
generate_class('other', csv_file_path, directory_path)


# test data
print("generating test data")

# wukuf
csv_file_path = 'test_data.csv'
directory_path = './data/test/wukuf'
generate_class('wukuf', csv_file_path, directory_path)

# rukuh
csv_file_path = 'test_data.csv'
directory_path = './data/test/rukuh'
generate_class('rukuh', csv_file_path, directory_path)

# sujud
csv_file_path = 'test_data.csv'
directory_path = './data/test/sujud'
generate_class('sujud', csv_file_path, directory_path)

# other
csv_file_path = 'test_data.csv'
directory_path = './data/test/other'
generate_class('other', csv_file_path, directory_path)