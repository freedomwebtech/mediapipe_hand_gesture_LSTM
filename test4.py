import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Prepare the mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Prepare MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load labels
labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Update according to your training labels
num_classes = len(labels_dict)

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

                # Get the confidence values
                confidence_values = prediction[0]
                confidence_percentage = confidence_values[predicted_index] * 100

                # Draw landmarks and connections using MediaPipe drawing utils
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

                # Draw a rectangle around the detected hand
                h, w, _ = frame.shape
                x_min = int(min(x_) * w)
                y_min = int(min(y_) * h)
                x_max = int(max(x_) * w)
                y_max = int(max(y_) * h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Draw the predicted character and confidence percentage
                cv2.putText(frame, f'{predicted_character}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Confidence: {confidence_percentage:.2f}%', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

                # Draw a confidence bar
                bar_length = 300  # Length of the confidence bar
                bar_height = 20   # Height of the confidence bar
                cv2.rectangle(frame, (50, 110), (50 + bar_length, 110 + bar_height), (0, 0, 0), -1)  # Background
                cv2.rectangle(frame, (50, 110), (50 + int(confidence_percentage * bar_length / 100), 110 + bar_height), (0, 255, 0), -1)  # Confidence bar

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
