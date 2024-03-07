import mediapipe as mp
import cv2

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def process_and_visualize_image(image_path):
    image = cv2.imread(image_path)
    # Convert the BGR image to RGB before processing.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect pose.
    results = pose.process(rgb_image)
    
    # Draw the pose annotations on the image.
    annotated_image = image.copy()
    if results.pose_landmarks:
        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    
    # Display the annotated image using OpenCV
    cv2.imshow('Pose Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = './data/train/wukuf/wukuf3.png'
process_and_visualize_image(image_path)
