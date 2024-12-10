

# **Sign Language Interpreter**

## **Overview**
This project is a **real-time Sign Language Interpreter** designed to detect and classify basic hand gestures using computer vision and machine learning techniques. The goal is to create an accessible tool for translating sign language gestures into text, contributing to accessibility and inclusivity.

This project was completed as a collaborative effort by **Evelyn Emeodume**, **Idongesit Utah**, and **Chimeziri Anyanwu**, in partial fulfillment of the requirements for our **Artificial Intelligence and Data Analysis** program at **Saskatchewan Polytechnic**.

---

## **Features**
- Real-time hand tracking and gesture detection using **MediaPipe**.
- Gesture classification using a trained **Convolutional Neural Network (CNN)**.
- Displays detected gestures on live video streams using **OpenCV**.
- Supports basic gestures like "Thumbs Up," "Victory Sign," and others.

---

## **Technologies Used**
- **Programming Language:** Python
- **Libraries and Tools:**
  - OpenCV: For video capture and real-time processing.
  - MediaPipe: For hand tracking and landmark detection.
  - TensorFlow/Keras: For training and deploying the machine learning model.
  - NumPy: For numerical operations.
  - Scikit-learn: For data preprocessing and model evaluation.

---

## **How It Works**
1. **Hand Tracking**:
   - Uses MediaPipe to detect hand landmarks in real time.
   - Captures positions of 21 hand landmarks for each frame.

2. **Gesture Classification**:
   - Preprocessed gesture data is fed into a CNN trained to classify gestures.
   - The output is displayed on the screen alongside the video feed.

3. **Real-Time Feedback**:
   - Provides visual feedback by overlaying the recognized gesture on the video feed.

---

## **Installation and Setup**
1. Clone this repository:
   ```bash
   git clone https://github.com/Mezirix/Sign-Language-Interpreter.git
   cd Sign-Language-Interpreter
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:
   ```bash
   python Real-Time Hand Tracking.py
   ```

---

## **Usage**
1. Start the program.
2. Position your hand in front of the webcam.
3. Perform gestures such as "Thumbs Up" or "Victory Sign."
4. The detected gesture will be displayed in real time.

---

## **Dataset**
Landmark data for gestures was collected manually using the **MediaPipe framework**.

---

## **Project Contributors**
- **Chimeziri Anyanwu**
- **Evelyn Emeodume**
- **Idongesit Utah**

---

## **Acknowledgments**
This project is part of the **Artificial Intelligence and Data Analysis** program at **Saskatchewan Polytechnic**. Special thanks to our instructors for their guidance and support.

---

## **Future Improvements**
- Expand the gesture set to include more sign language vocabulary.
- Integrate with speech synthesis to enable text-to-speech translation.
- Improve gesture recognition accuracy using larger datasets and advanced models.
- Explore deployment as a mobile or web application.

---

Feel free to suggest improvements or contribute by creating a pull request! 😊
