import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

# Initialize MediaPipe components for hand tracking.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the webcam capture using OpenCV.
cap = cv2.VideoCapture(0)

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

        # If hand landmarks are detected in the frame, draw the landmarks and hand connections.
        if results.multi_hand_landmarks:
            for landmarks, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Determine if the hand is left or right and assign a unique color.
                if hand.classification[0].label == "Left":
                    hand_color = (121, 22, 76)  # Color for the left hand

                    # Extract the landmark for the tip of the thumb (landmark index 4)
                    left_wrist = landmarks.landmark[0]
                    left_index_finger_mcp = landmarks.landmark[5]
                    left_pinky_mcp = landmarks.landmark[17]
                    left_thumb_tip = landmarks.landmark[4]

                    # Define the coordinates of reference point
                    refpt = np.array([left_thumb_tip.x, left_thumb_tip.y, left_thumb_tip.z])

                    # Get the coordinates of the wrist landmark.
                    x1, y1, z1 = left_wrist.x, left_wrist.y, left_wrist.z

                    # Get the coordinates of the index finger mcp landmark.
                    x2, y2, z2 = left_index_finger_mcp.x, left_index_finger_mcp.y, left_index_finger_mcp.z

                    # Get the coordinates of the pinky mcp landmark.
                    x3, y3, z3 = left_pinky_mcp.x, left_pinky_mcp.y, left_pinky_mcp.z

                    # Define the coordinates of three points
                    point1 = np.array([x1, y1, z1])
                    point2 = np.array([x2, y2, z2])
                    point3 = np.array([x3, y3, z3])

                    # Calculate the centroid of the triangle
                    centroid = (point1 + point2 + point3) / 3

                    # Calculate the normal vector of the plane
                    vector1 = point2 - point1
                    vector2 = point3 - point1
                    normal_vector = np.cross(vector1, vector2)

                    # Normalize the normal vector (optional)
                    normal_vector /= np.linalg.norm(normal_vector)

                    # Calculate the direction vector of the line perpendicular to the plane (parallel to the normal vector)
                    direction_vector = normal_vector

                    # Define the distance "k" from the centroid
                    k = 5.0 

                    # Calculate the coordinates of the point at distance "k" from the centroid along the line
                    point_above_plane = centroid + k * normal_vector
                    point_below_plane = centroid - k * normal_vector

                    # Calculate the distances from point1 to both points
                    distance_above = np.linalg.norm(point_above_plane - point1)
                    distance_below = np.linalg.norm(point_below_plane - point1)
                    
                    closer_point = None

                    # Determine which point is closer
                    if distance_above < distance_below:
                        closer_point = point_above_plane
                    else:
                        closer_point = point_below_plane

                    print("Coordinates of the closer point:", closer_point)


                else:
                    hand_color = (121, 44, 250)  # Color for the right hand

                # Draw landmarks and connections using the determined hand color.
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2))

        # Display the processed frame with the hand landmarks in a window named 'Hand Tracking'.
        cv2.imshow('Hand Tracking', image)

        # Check if the 'q' key is pressed to exit the loop and terminate the program.
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam capture when the loop is exited.
cap.release()

# Close all OpenCV windows.
cv2.destroyAllWindows()
