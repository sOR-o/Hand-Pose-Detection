import mediapipe as mp
import cv2
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

class VirtualCube:
    def __init__(self, initial_position=(320, 240, 0), size=40, speed=15):
        self.position = list(initial_position)
        self.size = size
        self.speed = speed

    def move(self, angle):
        self.position[0] += self.speed * np.cos(angle)
        self.position[1] += self.speed * np.sin(angle)
    
    def stop(self, andle):
        # stops the motion of the cube
        self.position[0] += 0
        self.position[1] += 0

virtual_cube = VirtualCube()

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

                    # Update the virtual cube's movement based on the angle
                    virtual_cube.move(angle)

                    # Define cube vertices
                    vertices = [
                        (int(virtual_cube.position[0] - virtual_cube.size / 2), int(virtual_cube.position[1] - virtual_cube.size / 2), int(virtual_cube.position[2] - virtual_cube.size / 2)),
                        (int(virtual_cube.position[0] + virtual_cube.size / 2), int(virtual_cube.position[1] - virtual_cube.size / 2), int(virtual_cube.position[2] - virtual_cube.size / 2)),
                        (int(virtual_cube.position[0] + virtual_cube.size / 2), int(virtual_cube.position[1] + virtual_cube.size / 2), int(virtual_cube.position[2] - virtual_cube.size / 2)),
                        (int(virtual_cube.position[0] - virtual_cube.size / 2), int(virtual_cube.position[1] + virtual_cube.size / 2), int(virtual_cube.position[2] - virtual_cube.size / 2)),
                        (int(virtual_cube.position[0] - virtual_cube.size / 2), int(virtual_cube.position[1] - virtual_cube.size / 2), int(virtual_cube.position[2] + virtual_cube.size / 2)),
                        (int(virtual_cube.position[0] + virtual_cube.size / 2), int(virtual_cube.position[1] - virtual_cube.size / 2), int(virtual_cube.position[2] + virtual_cube.size / 2)),
                        (int(virtual_cube.position[0] + virtual_cube.size / 2), int(virtual_cube.position[1] + virtual_cube.size / 2), int(virtual_cube.position[2] + virtual_cube.size / 2)),
                        (int(virtual_cube.position[0] - virtual_cube.size / 2), int(virtual_cube.position[1] + virtual_cube.size / 2), int(virtual_cube.position[2] + virtual_cube.size / 2))
                    ]

                    # Define cube edges
                    edges = [
                        [vertices[0], vertices[1], vertices[2], vertices[3]],
                        [vertices[4], vertices[5], vertices[6], vertices[7]],
                        [vertices[0], vertices[1], vertices[5], vertices[4]],
                        [vertices[2], vertices[3], vertices[7], vertices[6]],
                        [vertices[1], vertices[2], vertices[6], vertices[5]],
                        [vertices[4], vertices[7], vertices[3], vertices[0]]
                    ]

                    # Draw the cube
                    for i, edge in enumerate(edges):
                        # Add shading effect by varying line color and thickness
                        line_color = (0, 255 - i * 10, 0)  # Gradually change the color from bright to dark
                        line_thickness = 2 + i  # Gradually increase line thickness
                        cv2.line(image, (edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]), line_color, line_thickness)


                # Determine if the hand is left or right and assign a unique color.
                if hand.classification[0].label == "Left":
                    hand_color = (121, 22, 76)  # Color for the left hand
                else:
                    hand_color = (121, 44, 250)  # Color for the right hand

                # Draw landmarks and connections using the determined hand color for both hands
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2))

        cv2.imshow('Hand Tracking with 3D Cube', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
