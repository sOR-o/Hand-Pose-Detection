# Hand-Pose-Estimation
This is a comprehensive sign language translation system that integrates hand pose estimation, language modeling, and real-time translation capabilities. The system comprises several components, including scripts for generating custom hand pose labeled data, model training scripts, a Streamlit application for real-time translation, and a server component for processing sign language sequences using a language model. Leveraging an awq quantized variant of Llama-2-13b accelerated by llamacpp for GPU inference, the system achieves efficient and accurate translation of sign language gestures into coherent sentences. The carefully crafted prompt template ensures the accuracy and fluency of translations, making the system suitable for diverse communication scenarios.

[<img src="./assets/asset04.png" height=400 weidth=600></img>](https://github.com/sOR-o/Hand-Pose-Estimation/assets/69918938/4fc4ee75-e6fd-4d47-8b89-e921ec92cdb5)


## Dataset Generation and Custom Gesture Detection
This project offers a versatile platform for hand-related tasks, including `dataset generation` and custom hand gesture detection. The best place to start and learn about the project's evolution is the ["learning/testing"](https://github.com/sOR-o/Hand-Pose-Estimation/tree/main/learning/testing) folder. Here, you'll find different levels of understanding, providing comprehensive insights into the project's development.

###### One of the Application ↓
# Real-time Sign Language Translator

[<img src="./assets/asset04.png" height=400 weidth=600></img>](https://github.com/sOR-o/Hand-Pose-Estimation/assets/69918938/1a7b84b4-715a-41e0-847a-4ab638ff66e2)

One of the key applications of hand pose estimation is sign language recognition. This project provides tools and resources to develop and train models for recognizing sign language gestures. Here's how it works:

- **Hand Gesture Detection:** The system detects and tracks hand gestures using hand pose estimation techniques.

- **Predicting Labels:** After detecting the hand gestures, the system captures the predicted labels associated with each gesture.

- **Storing Predicted Labels:** The predicted labels, representing sign language gestures, are stored for interpretation and translation.

- **Real-Time Translation:** The stored predicted labels are passed into a LLM, i.e `Llama2 13B` for translation. This model interprets the sign language gestures and generates the corresponding text.

- **Displaying Real-Time Translation:** The translated text is then displayed in real-time, allowing seamless communication between sign language users and non-signers.

## Getting Started

1. **Clone the Repository**: `git clone https://github.com/sOR-o/Hand-Pose-Estimation.git`
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run LlamaCPP Setup Script**: `bash setup_llamacpp.sh`
4. **Start LlamaCPP Server**: `bash start_llamacpp_server.sh`
5. **Run the Model Script**: `python streamlit/sign-language/Main/model.py`
6. **Launch Frame Interface**: `streamlit run streamlit/sign-language/Main/frame.py`



###### All set! To add more custom gestures, check out the [learning/testing](https://github.com/sOR-o/Hand-Pose-Estimation/tree/main/learning/testing) directory.


## Customization Options
This project is designed to be flexible and easily customizable. Here are some aspects you can modify:

- **Custom Hand Gestures:** Add more custom hand gestures based on your specific use case. Explore the ["learning/testing"](https://github.com/sOR-o/Hand-Pose-Estimation/tree/main/learning/testing) folder for examples and adapt them to your needs.

- **Color and Thickness of Hand Markings:** Tailor the visual appearance by changing the color and thickness of hand markings. Explore the code related to drawing hand landmarks and adjust parameters to suit your preferences.

- **Hand Tracking Drawing:** You can disable the drawing of hand tracking lines or modify the visualization according to your project requirements. This can be useful if you want to integrate the hand pose estimation into a different visualization context.

- **Prediction Integration:** Utilize the hand pose predictions in a way that fits your application. Extract the hand pose information and integrate it into your broader project for a seamless user experience.

-Can be improved by transfer learning (obviously 😉)

## Contributing

Contributions to this project are welcome! Whether it's bug fixes, new features, or documentation improvements, your contributions are valuable.
