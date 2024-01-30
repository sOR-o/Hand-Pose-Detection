# image_predictor.py

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pickle

# Load the model
model = load_model("streamlit/model/keras_model.h5", compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the labels
class_names = open("streamlit/model/labels.txt", "r").readlines()

def predict_image(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return {
        "class_name": class_name,
        "confidence_score": confidence_score
    }

# Save the function as a pickled file only when the script is run directly
if __name__ == "__main__":
    with open("streamlit/image_predictor.pkl", "wb") as file:
        pickle.dump(predict_image, file)
