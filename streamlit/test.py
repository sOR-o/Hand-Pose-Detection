import mediapipe as mp
import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# Set Streamlit page configuration
st.set_page_config(page_title="hand-pose-detection", page_icon="assets/logo.png")

# Load media pipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load hand gesture classifier
classifier = Classifier("streamlit/model/keras_model.h5", "streamlit/model/labels.txt")

# Define gesture labels and confidence threshold
labels = ['chill', 'suchuu', 'hey, hello', 'kwiks', 'peace out', 'stop', 'thumbs up']
confidence_threshold = 65.00

# Initialize hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1)

# Set hand to detect (left or right)
hand_to_detect = "left"

# Hand cropping constants
offset = 30  # Adjust this offset as needed
imgSize = 300

# Title for Streamlit app
st.title("Hand Pose Detection")

def video_frame_callback(frame):
        # Convert the frame to opencv format
        image = frame.to_ndarray(format="bgr24")

        # Process the frame using mediapipe hands
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Create a black canvas
        canvas = np.zeros_like(image)

        try:
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
                                                  mp_drawing.DrawingSpec(color=hand_color, thickness=1, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=hand_color, thickness=1, circle_radius=1))

                        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=hand_color, thickness=1, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=hand_color, thickness=1, circle_radius=1))
                        
                        # Hand cropping
                        min_x = min(int(lm.x * canvas.shape[1]) for lm in landmarks.landmark)
                        max_x = max(int(lm.x * canvas.shape[1]) for lm in landmarks.landmark)
                        min_y = min(int(lm.y * canvas.shape[0]) for lm in landmarks.landmark)
                        max_y = max(int(lm.y * canvas.shape[0]) for lm in landmarks.landmark)

                        x, y, w, h = min_x, min_y, max_x - min_x, max_y - min_y

                        # Check if the region of interest is too small
                        if w >= 10 and h >= 10:
                            imgCrop = canvas[y - offset:y + h + offset, x - offset:x + w + offset]

                            # Resize the cropped image to 400x400
                            imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

                            # Display the cropped hand lines when the 's' key is pressed
                            prediction, index = classifier.getPrediction(imgCrop)

                            # Get the index of the maximum confidence score and getting the corresponding label
                            predicted_index = np.argmax(prediction)
                            predicted_label = labels[predicted_index]
                            confidence_percentage = prediction[predicted_index] * 100

                            if prediction[predicted_index] * 100 >= confidence_threshold:
                                label_text = f'Predicted Label: "{predicted_label}" Confidence: {confidence_percentage:.2f}%'
                                cv2.putText(image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (121, 22, 76), 1, cv2.LINE_AA)

                            else:
                                label_text = "gesture not recognized or confidence too low."
                                cv2.putText(image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (121, 44, 250), 1,
                                            cv2.LINE_AA)


        except Exception as e:
            # Handle any exception and display the error message on the frame
            error_text = "make sure your hand is in the frame."
            cv2.putText(image, error_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


        return av.VideoFrame.from_ndarray(image, format='bgr24')

webrtc_streamer(key="key", video_frame_callback=video_frame_callback, rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

# Display the gesture images in a horizontal line
gesture_images = [
    "learning/testing/data/Data/chill/Image_1706386389.546958.jpg",
    "learning/testing/data/Data/chuu/Image_1706373053.875865.jpg",
    "learning/testing/data/Data/hi/Image_1706372404.508589.jpg",
    "learning/testing/data/Data/kwiks/Image_1706387011.21945.jpg",
    "learning/testing/data/Data/peace/Image_1706372568.933511.jpg",
    "learning/testing/data/Data/stop/Image_1706372696.49316.jpg",
    "learning/testing/data/Data/thumbsUp/Image_1706372137.717592.jpg"
]

# Display the images horizontally
st.image(gesture_images, width=100, caption=['chill', 'suchuu', 'hey, hello', 'kwiks', 'peace out', 'stop', 'thumbs up'])

# App description
st.markdown("""

    This hand pose detection model has been trained on a limited dataset for demonstration purposes.
    It may not cover all possible gestures and is not fully fine-tuned, serving primarily as an illustrative example.

    If you are interested in exploring custom gestures or fine-tuning the model, you are welcome to visit my 
    [GitHub](https://github.com/sOR-o/Hand-Pose-Estimation) for more details, resources, and the opportunity to contribute.
    Feel free to experiment, enhance, and customize the model based on your requirements.
            
    Note: The dimensions of the frames and the dataset on which it is trained may differ from what is used in this demo.

    -Can be improved by transfer learning (obviously 😉)
""")



