# testing02.py

import pickle

print("Before loading pickled function.")

# Load the pickled function
pickle_in =  open("streamlit/image_predictor.pkl", "rb")
predict_image = pickle.load(pickle_in)

print("After loading pickled function.")

# Provide the path to the image you want to predict
image_path = "learning/testing/data/Data/chill/Image_1706386389.953801.jpg"

# Call the loaded function
result = predict_image(image_path)

# Access the prediction outputs
class_name = result["class_name"]
confidence_score = result["confidence_score"]

# Print or use the results as needed
print("Predicted Class:", class_name)
print("Confidence Score:", confidence_score)
