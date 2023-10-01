import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

class VirtualBall:
    def __init__(self, initial_position=(320, 240), radius=20, speed=15):
        self.position = list(initial_position)
        self.radius = radius
        self.speed = speed

    def move(self, angle):
        self.position[0] += self.speed * np.cos(angle)
        self.position[1] += self.speed * np.sin(angle)

virtual_ball = VirtualBall()

with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for landmarks, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                if (hand.classification[0].label == "Right") or (hand.classification[0].label == "Left"):

                    index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_finger_base = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                    # Calculate the angle between the index finger base and tip
                    angle = np.arctan2(index_finger_tip.y - index_finger_base.y,
                                       index_finger_tip.x - index_finger_base.x)

                    # Update the virtual ball's movement based on the angle
                    virtual_ball.move(angle)

                    cv2.circle(image, (int(virtual_ball.position[0]), int(virtual_ball.position[1])),
                               virtual_ball.radius, (0, 255, 0), -1)

                # Determine if the hand is left or right and assign a unique color.
                if hand.classification[0].label == "Left":
                    hand_color = (121, 22, 76)  # Color for the left hand
                else:
                    hand_color = (121, 44, 250)  # Color for the right hand

                # Draw landmarks and connections using the determined hand color for both hands
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2))

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
