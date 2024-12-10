import cv2
import mediapipe as mp
import numpy as np


def recognize_gesture(hand_landmarks):
    # Get key landmarks for fingers
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP]

    # Gesture: Thumbs Up
    if thumb_tip.y < thumb_ip.y and abs(thumb_tip.x - index_tip.x) > 0.1:
        return "Thumbs Up"

    # Gesture: Five Fingers (Open Palm)
    fingers_open = [
        thumb_tip.y < thumb_ip.y,
        index_tip.y < index_pip.y,
        middle_tip.y < middle_pip.y,
        pinky_tip.y < pinky_pip.y
    ]
    if all(fingers_open):
        return "Five Fingers"

    # Gesture: Pinky Finger
    if pinky_tip.y < pinky_pip.y and all([
        index_tip.y > index_pip.y,
        middle_tip.y > middle_pip.y,
    ]):
        return "Pinky Finger"

    # Gesture: Middle Finger
    if middle_tip.y < middle_pip.y and all([
        index_tip.y > index_pip.y,
        pinky_tip.y > pinky_pip.y
    ]):
        return "Middle Finger"

    # Gesture: Index Finger
    if index_tip.y < index_pip.y and all([
        middle_tip.y > middle_pip.y,
        pinky_tip.y > pinky_pip.y
    ]):
        return "Index Finger"

    return "Unknown Gesture"


def process_video():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize OpenCV webcam capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Process and annotate the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks)

                # Display the recognized gesture
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Sign Language Interpreter", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()
