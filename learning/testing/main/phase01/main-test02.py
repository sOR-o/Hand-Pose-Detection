import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe components for hand tracking.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the webcam capture using OpenCV.
cap = cv2.VideoCapture(0)

# Load or define a dictionary mapping gestures to words.
gesture_word_mapping = {
    "thumbs_up": "Hello",
    "index_finger_up": "Good",
    "pinky_finger_up": "Bye",
    # Add more gestures and corresponding words as needed
}

# Create a loop to continuously process frames from the webcam.
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        # Capture a frame from the webcam.
        ret, frame = cap.read()

        # Convert the frame from BGR to RGB color space (required by MediaPipe).
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the RGB image with the MediaPipe 'hands' model to detect and track hands.
        results = hands.process(image)
        image.flags.writeable = True

        # Convert the processed image back to BGR color space.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        recognized_words = []

        # If hand landmarks are detected in the frame, draw the landmarks and hand connections.
        if results.multi_hand_landmarks:
            for landmarks, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Extract landmarks for specific fingers (e.g., thumb, index finger, pinky).
                # Modify this part based on your gesture recognition model or method.

                # Recognize the gesture based on the landmarks.
                # You may replace this with your own gesture recognition logic.
                # Here, we use a simple example by checking if the thumb is up.
                thumb_tip = landmarks.landmark[4]
                index_finger_tip = landmarks.landmark[8]
                pinky_tip = landmarks.landmark[20]

                # Add more conditions to recognize different gestures.
                if thumb_tip.y > index_finger_tip.y and thumb_tip.y > pinky_tip.y:
                    recognized_words.append(gesture_word_mapping.get("thumbs_up"))

                # Similar logic for the right hand.
                # Draw landmarks and connections using the determined hand color.
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2))

        # Display the processed frame with the hand landmarks in a window named 'Hand Tracking'.
        cv2.imshow('Hand Tracking', image)

        # Print the recognized words.
        if recognized_words:
            print("Recognized Words:", recognized_words)

        # Check if the 'q' key is pressed to exit the loop and terminate the program.
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam capture when the loop is exited.
cap.release()

# Close all OpenCV windows.
cv2.destroyAllWindows()
