import mediapipe as mp
import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
import streamlit as st
import time
import json
import requests
from streamlit_lottie import st_lottie
from datetime import datetime
import functions

st.set_page_config(page_title="sign-language", page_icon="assets/asset08.png")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

classifier = Classifier("streamlit/sign-language/model02/keras_model.h5", "streamlit/sign-language/model02/labels.txt")

labels = ['bathroom', 'bye', 'bye', 'doing', 'fine', 'fine', 'good', 'hello', 'hello', 'help', 'how', 'live', 'love', 'me', 'me', 'very much', 'name', 'question mark', 'stop', 'thankyou', 'thankyou', 'thumbsup', 'thumbsup', 'what', 'where', 'you','are']
confidence_threshold = 70.00

# Initialize hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=2)

# Hand cropping constants
offset = 30  # Adjust this offset as needed
imgSizex = 600
imgSizey = 600

brokenFilepath = "streamlit/sign-language/Main/broken.txt"
correctFilepath = "streamlit/sign-language/Main/correct.txt"
functions.clear_file_content(brokenFilepath)
functions.clear_file_content(correctFilepath)

st.title("sign-language")

# Create a placeholder for displaying the webcam feed
image_placeholder = st.image([])

def load_lottie_url(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)

lottie = load_lottie_url("streamlit/sign-language/processing.json")

# Display "processing" text
st.write("**Processing** ", end=" ")

# Display animation
if lottie is not None:
    st_lottie(lottie, speed=2, width=75, height=75, key="initial")

# Initial text content
text_container = st.empty()
one_text = ""
text_content = ""
input_text = []

current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Display the gesture images in a horizontal line
gesture_images = [
    "learning/testing/data/sign-language/bye2/Image_1707039809.9438741.jpg",
    "learning/testing/data/sign-language/fine2/Image_1707040085.656412.jpg",
    "learning/testing/data/sign-language/hello2/Image_1707041322.978965.jpg",
    "learning/testing/data/sign-language/good/Image_1707040603.533092.jpg",
    "learning/testing/data/sign-language/help/Image_1707044857.938269.jpg",
    "learning/testing/data/sign-language/how/Image_1707045619.622827.jpg",
    "learning/testing/data/sign-language/live/Image_1707044991.304802.jpg"
]

# Display the images horizontally
st.image(gesture_images, width=100, caption=['bye', 'fine', 'hello', 'good', 'help', 'how', 'live'])


# Display the gesture images in a horizontal line
gesture_images = [
    "learning/testing/data/sign-language/love/Image_1707045538.722445.jpg",
    "learning/testing/data/sign-language/me2/Image_1707041555.539006.jpg",
    "learning/testing/data/sign-language/much/Image_1707045368.705156.jpg",
    "learning/testing/data/sign-language/name/Image_1707045245.527601.jpg",
    "learning/testing/data/sign-language/ques/Image_1707041858.998496.jpg",
    "learning/testing/data/sign-language/stop/Image_1707042065.17011.jpg",
    "learning/testing/data/sign-language/thankyou2/Image_1707042271.704127.jpg"
]

# Display the images horizontally
st.image(gesture_images, width=100, caption=['love', 'me', 'much', 'name', 'ques', 'stop', 'thanku'])


# Display the gesture images in a horizontal line
gesture_images = [
    "learning/testing/data/sign-language/thumbsup2/Image_1707042490.842237.jpg",
    "learning/testing/data/sign-language/what/Image_1707043113.307008.jpg",
    "learning/testing/data/sign-language/where/Image_1707042587.2094588.jpg",
    "learning/testing/data/sign-language/you/Image_1707042701.319511.jpg",
    "learning/testing/data/sign-language/bathroom/Image_1707039429.5277262.jpg",
    "learning/testing/data/sign-language/doing/Image_1707045801.082792.jpg",
    "learning/testing/data/sign-language/r/Image_1707055225.409577.jpg"
]

# Display the images horizontally
st.image(gesture_images, width=100, caption=['thumbsup', 'what', 'where', 'you','bathroom', 'doing', 'r'])

st.markdown("""
    <div style="text-align: center;">

    This hand pose detection model has been trained on a limited dataset for demonstration purposes. It may not cover all possible gestures and is not fully fine-tuned, serving primarily as an illustrative example.

    If you are interested in exploring custom gestures or fine-tuning the model, you are welcome to visit my [GitHub ](https://github.com/sOR-o/Hand-Pose-Estimation) for more details, resources, and the opportunity to contribute. Feel free to experiment, enhance, and customize the model based on your requirements.

    -Can be improved by transfer learning (obviously 😉)

    </div>
""", unsafe_allow_html=True)

camera_active = True  # Flag to track camera state

imgCrop = None
no_frame = 0

while True:
    # Check if the 'ESC' key is pressed to toggle the camera state
    key = cv2.waitKey(1)
    if key == 27:  # ASCII code for 'ESC'
        camera_active = not camera_active

    if camera_active:
        # Initialize webcam if not already done
        cap = cv2.VideoCapture(0)

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

        # Reset flags for each iteration
        left_hand_detected = False
        right_hand_detected = False

        first_line = functions.read_and_delete_first_line(correctFilepath)
        if first_line is not None:
            first_line = first_line.split(" : ")[1]

            with text_container:
                text_content += "convo : " + first_line + "\n"
                st.text_area("Real-Time Text", text_content, height=150, max_chars=None, key=None)

        
        try:
            if results.multi_hand_landmarks:

                # Initialize lists to store coordinates of each hand's landmarks
                left_hand_landmarks = []
                right_hand_landmarks = []

                for landmarks, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Determine if the hand is left or right and assign a unique color.
                    if hand.classification[0].label == "Left":
                        hand_color = (121, 22, 76)  # Color for the left hand
                        left_hand_detected = True
                    else:
                        hand_color = (121, 44, 250)  # Color for the right hand
                        right_hand_detected = True

                    # Draw landmarks and connections using the determined hand color for both hands
                    mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2))

                    # Draw landmarks and connections using the determined hand color for both hands
                    mp_drawing.draw_landmarks(canvas, landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2))

                    # Append landmarks coordinates to the respective hand list
                    if hand.classification[0].label == "Left":
                        left_hand_landmarks.extend([(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in landmarks.landmark])
                    else:
                        right_hand_landmarks.extend([(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in landmarks.landmark])

                    # Hand cropping
                    if hand.classification[0].label == "Left":
                        left_hand_landmarks.extend([(lm.x * canvas.shape[1], lm.y * canvas.shape[0]) for lm in landmarks.landmark])
                    else:
                        right_hand_landmarks.extend([(lm.x * canvas.shape[1], lm.y * canvas.shape[0]) for lm in landmarks.landmark])

                    # Hand cropping
                    min_x = min(int(lm.x * canvas.shape[1]) for lm in landmarks.landmark)
                    max_x = max(int(lm.x * canvas.shape[1]) for lm in landmarks.landmark)
                    min_y = min(int(lm.y * canvas.shape[0]) for lm in landmarks.landmark)
                    max_y = max(int(lm.y * canvas.shape[0]) for lm in landmarks.landmark)

                    x, y, w, h = min_x, min_y, max_x - min_x, max_y - min_y
                    imgCrop = canvas[y - offset:y + h + offset, x - offset:x + w + offset]

                    # Resize the cropped image to imgSize x imgSize
                    imgCrop = cv2.resize(imgCrop, (imgSizex, imgSizey))

                # Check if both hands are close
                if left_hand_detected and right_hand_detected:
                    # Create a rectangle that encloses both hands
                    min_x = min(left_hand_landmarks + right_hand_landmarks, key=lambda item: item[0])[0]
                    min_y = min(left_hand_landmarks + right_hand_landmarks, key=lambda item: item[1])[1]
                    max_x = max(left_hand_landmarks + right_hand_landmarks, key=lambda item: item[0])[0]
                    max_y = max(left_hand_landmarks + right_hand_landmarks, key=lambda item: item[1])[1]

                    # Add offset to the rectangle
                    min_x -= offset
                    min_y -= offset
                    max_x += offset
                    max_y += offset

                    # Crop the region enclosed by the modified rectangle
                    imgCrop = canvas[int(min_y):int(max_y), int(min_x):int(max_x)]

                # Resize the cropped image to imgSizex x imgSizey
                imgCrop = cv2.resize(imgCrop, (imgSizex, imgSizey))

                # Display the cropped hand lines when the 's' key is pressed
                prediction, index = classifier.getPrediction(imgCrop)

                # Get the index of the maximum confidence score and getting the corresponding label
                predicted_index = np.argmax(prediction)
                predicted_label = labels[predicted_index]

                if prediction[predicted_index] * 100 >= confidence_threshold:
                    label_text = f'Predicted Label: "{predicted_label}" Confidence: {prediction[predicted_index] * 100:.2f}%'
                    cv2.putText(image, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (121, 22, 76), 2, cv2.LINE_AA)

                    if len(input_text) == 0 or input_text[-1] != predicted_label:
                        input_text.append(predicted_label)
                    
                    
                else:
                    label_text = "Gesture not recognized or confidence too low."
                    cv2.putText(image, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (121, 44, 250), 2, cv2.LINE_AA)
                    

            else:
                if len(input_text) > 0:
                    one_text = ""
                    for i in input_text:
                        one_text += i + " "

                    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    functions.add_string_to_file(one_text, brokenFilepath)

                    input_text.clear()          

        except Exception as e:
            # Handle any exception and display the error message on the frame
            error_text = "Make sure your hands are in the frame."
            cv2.putText(image, error_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame with hand tracking
        image_placeholder.image(image, channels="BGR")

    else:
        # Stop the camera and display a black image
        cap.release()
        image_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")

    # Exit the loop when the 'ESC' key is pressed
    if key == 27:
        break

# Release resources
if cap:
    cap.release()
cv2.destroyAllWindows()
