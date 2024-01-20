import mediapipe as mp
import cv2
import numpy as np
import math
from cvzone.ClassificationModule import Classifier  # Make sure you have this module

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=2)

# Load the classifier model and labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Hand cropping constants
offset = 30  # Adjust this offset as needed
imgSize = 400

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame using mediapipe hands
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for landmarks, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Determine if the hand is left or right and assign a unique color.
            if hand.classification[0].label == "Left":
                hand_color = (121, 22, 76)  # Color for the left hand
            else:
                hand_color = (121, 44, 250)  # Color for the right hand

            # Draw landmarks and connections using the determined hand color for both hands
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2))

            # Hand cropping (without saving)
            min_x = min(int(lm.x * image.shape[1]) for lm in landmarks.landmark)
            max_x = max(int(lm.x * image.shape[1]) for lm in landmarks.landmark)
            min_y = min(int(lm.y * image.shape[0]) for lm in landmarks.landmark)
            max_y = max(int(lm.y * image.shape[0]) for lm in landmarks.landmark)

            x, y, w, h = min_x, min_y, max_x - min_x, max_y - min_y
            imgCrop = image[y - offset:y + h + offset, x - offset:x + w + offset]

            # Display the cropped hand image when the 's' key is pressed
            cv2.imshow("ImageCrop", imgCrop)

            # Convert the cropped image to RGB and resize it to match the model input shape
            imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            imgCrop = cv2.resize(imgCrop, (224, 224))  # Adjust the size according to your model

            # Normalize the image
            imgCrop = imgCrop / 255.0

            # Add batch dimension for model prediction
            imgCrop = np.expand_dims(imgCrop, axis=0)

            # Predict the gesture using the classifier
            prediction, index = classifier.getPrediction(imgCrop)
            print(f"Predicted Gesture: {prediction}, Index: {index}")

    # Display the frame with hand tracking
    cv2.imshow('Hand Tracking', image)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
