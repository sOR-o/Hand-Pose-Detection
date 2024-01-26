import mediapipe as mp
import cv2
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1)

# Initialize webcam
cap = cv2.VideoCapture(0)

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

# Hand cropping constants
offset = 30  # Adjust this offset as needed
imgSize = 300
folder = "Data/three"
counter = 0

# Create the "Data" folder if it doesn't exist
if not cv2.os.path.exists(folder):
    cv2.os.makedirs(folder)

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

    # Create a black canvas
    canvas = np.zeros_like(image)

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
                mp_drawing.draw_landmarks(canvas, landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2))

                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2))

                # Hand cropping
                min_x = min(int(lm.x * canvas.shape[1]) for lm in landmarks.landmark)
                max_x = max(int(lm.x * canvas.shape[1]) for lm in landmarks.landmark)
                min_y = min(int(lm.y * canvas.shape[0]) for lm in landmarks.landmark)
                max_y = max(int(lm.y * canvas.shape[0]) for lm in landmarks.landmark)

                x, y, w, h = min_x, min_y, max_x - min_x, max_y - min_y
                imgCrop = canvas[y - offset:y + h + offset, x - offset:x + w + offset]

                # Resize the cropped image to 400x400
                imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

                # Display the cropped hand lines when the 's' key is pressed
                cv2.imshow("HandLinesCrop", imgCrop)

                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    file_path = cv2.os.path.join(folder, f'Image_{time.time()}.jpg')
                    cv2.imwrite(file_path, imgCrop)
                    print(f"Hand lines saved: {file_path} ({counter})")

    # Display the frame with hand tracking
    cv2.imshow('Hand Tracking', image)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
