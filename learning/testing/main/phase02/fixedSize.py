import mediapipe as mp
import cv2
import numpy as np
import math
from cvzone.ClassificationModule import Classifier

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

classifier = Classifier("model/keras_model.h5", "model/labels.txt")

# Initialize hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Hand cropping constants
offset = 30  # Adjust this offset as needed
imgSize = 400

# Ask the user which hand to detect
hand_to_detect = input("Enter the hand to detect (left/right): ").lower()
if hand_to_detect not in ['left', 'right']:
    print("Invalid input. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()


# For mirror image
if hand_to_detect == "left":
    hand_to_detect = "right"
else:
    hand_to_detect = "left"


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
            detected_hand = hand.classification[0].label.lower()
            if detected_hand == hand_to_detect:
                if detected_hand == "left":
                    hand_color = (121, 44, 250)  # Color for the left hand
                else:
                    hand_color = (121, 22, 76)  # Color for the right hand

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
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = image[y - offset:y + h + offset, x - offset:x + w + offset]

                # Resize the cropped image to 400x400
                imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

                cv2.imshow("ImageCrop", imgCrop)

                prediction, index = classifier.getPrediction(imgCrop)
                print(prediction, index)

    # Display the frame with hand tracking
    cv2.imshow('Hand Tracking', image)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()