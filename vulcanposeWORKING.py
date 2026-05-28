import cv2
import mediapipe as mp
import numpy as np 
import csv

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

vulcan_img = cv2.imread("vulcan.png")
filename="vulcan_gesture_data.csv"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
raw=[]
np.savez('gestures.npz', features=[f'{axis}{i}' for i in range(1, 22) for axis in ('x', 'y', 'z')], labels=['label'])
while True:
    success, frame = cap.read()
    if not success:
        break

    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_frame)
    
    key = cv2.waitKey(1) & 0xFF

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            
            if key == ord('a'):
                wrist = handLms.landmark[mp_hands.HandLandmark.WRIST]
                index_mcp = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                
                dist = np.linalg.norm(np.array([index_mcp.x, index_mcp.y]) - np.array([wrist.x, wrist.y]))
                if dist == 0: 
                    dist = 0.001

                hand_features = []
                for id, lm in enumerate(handLms.landmark):
                    lm_x = float((lm.x - wrist.x) / dist)
                    lm_y = float((lm.y - wrist.y) / dist)
                    lm_z = float((lm.z - wrist.z) / dist)
                    
                    hand_features.append([lm_x, lm_y, lm_z])
                hand_features_flat = [coord for point in hand_features for coord in point]
                raw.append(hand_features_flat)

    cv2.imshow("Frame", frame)
    
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

np.savez('gestures.npz', features=raw, labels=[i for i in range(0, len(raw))])
#print(flat)