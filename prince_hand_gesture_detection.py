

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Gesture dictionary
gestures = {
    0: 'Fist',
    1: 'Open Hand',
    2: 'Peace',
    3: 'Thumbs Up',
    4: 'Thumbs Down'
}

def detect_gesture(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        # Assuming 4th landmark is the tip of the index finger
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        # Assuming 8th landmark is the tip of the thumb
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        
        # Calculate distance between index finger tip and thumb tip
        distance = ((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)**0.5
        
        # Classify gesture based on distance
        if distance < 0.1:
            return 0  # Fist
        elif distance > 0.25:
            return 1  # Open Hand
        elif index_tip.y < thumb_tip.y:
            return 2  # Peace
        elif index_tip.y > thumb_tip.y:
            return 3  # Thumbs Up
        else:
            return 4  # Thumbs Down
    else:
        return None

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for natural view
    frame = cv2.flip(frame, 1)
    
    # Detect gesture
    gesture_id = detect_gesture(frame)
    if gesture_id is not None:
        gesture = gestures[gesture_id]
        cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
