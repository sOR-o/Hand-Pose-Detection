import cv2
import numpy as np
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Create a virtual ball object
class VirtualBall:
    def __init__(self, initial_position=(0, 0, -5), radius=0.1, speed=0.05):
        self.position = list(initial_position)
        self.radius = radius
        self.speed = speed

    def move_up(self):
        self.position[1] += self.speed

    def move_down(self):
        self.position[1] -= self.speed

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# Initialize pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((640, 480), DOUBLEBUF | OPENGL)
pygame.display.set_caption('Virtual Ball')

# Create a virtual ball
virtual_ball = VirtualBall()

# Main loop
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                exit()

        # Capture video frame
        ret, frame = cv2.VideoCapture(0).read()
        image, results = mediapipe_detection(frame, holistic)

        # Check if right hand landmarks were detected
        if results.right_hand_landmarks is not None:
            keypoints = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()

            # Update virtual ball position based on hand gestures
            if keypoints[8 * 3 + 1] < keypoints[0 * 3 + 1]:
                virtual_ball.move_up()
            elif keypoints[8 * 3 + 1] > keypoints[0 * 3 + 1]:
                virtual_ball.move_down()

        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up the virtual camera
        gluPerspective(45, (640 / 480), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

        # Render the 3D sphere (ball)
        glTranslatef(*virtual_ball.position)
        glColor3f(0.0, 1.0, 0.0)
        glutSolidSphere(virtual_ball.radius, 32, 32)

        # Swap the display buffers
        pygame.display.flip()
