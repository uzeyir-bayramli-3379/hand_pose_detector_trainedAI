import cv2
import mediapipe as mp
import numpy as np 
import csv
import torch
import torch.nn as nn
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
class TestModel(nn.Module):
    def __init__(self):

        super(TestModel, self).__init__()

        self.input_layer = nn.Linear(63,64)
        self.relu=nn.ReLU()
        self.linear = nn.Linear(64, 3)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.linear(x)
        return x
model=TestModel()
model.load_state_dict(torch.load('vulcan_gesture_model.pth', weights_only=True))
model.eval()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
raw=[]
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
            features_tensor = torch.tensor(hand_features_flat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs = model(features_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
                probs = torch.softmax(outputs, dim=1)
                confidence = probs[0][predicted_class].item()
            if confidence > 0.75:
                cv2.putText(frame, f'Class: {predicted_class} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
