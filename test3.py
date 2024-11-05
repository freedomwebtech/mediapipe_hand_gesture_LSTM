import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Prepare the mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load labels
labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Update according to your training labels
# If using LabelEncoder, you can load it here if saved

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            # Extract landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the data (if required, depending on how you trained your model)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalization example
                data_aux.append(y - min(y_))

            # Reshape the input for LSTM: (1, timesteps, features)
            if len(data_aux) == 42:  # Ensure it matches your feature count
                input_data = np.array(data_aux).reshape((1, 6, 7))  # Adjust according to your setup
                # Make prediction
                prediction = model.predict(input_data)
                predicted_index = np.argmax(prediction)
                predicted_character = labels_dict[predicted_index]

                # Draw the predictions on the frame
                cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
