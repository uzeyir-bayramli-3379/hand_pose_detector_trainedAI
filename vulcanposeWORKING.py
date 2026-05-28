import cv2
import mediapipe as mp
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
vulcan_img=cv2.imread("vulcan.png")
mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
while True:
    success,frame=cap.read()
    if success:
        RGB_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=hands.process(RGB_frame)
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                #print(handLms)
                mp_drawing.draw_landmarks(frame,handLms,mp_hands.HAND_CONNECTIONS)
                height, width, _ = RGB_frame.shape
                # Draw line between index finger tip and thumb tip
                #index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                #pt1 = (int(index_tip.x * width), int(index_tip.y * height))
                #thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
                #pt2 = (int(thumb_tip.x * width), int(thumb_tip.y * height))
                #cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = handLms.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = handLms.landmark[mp_hands.HandLandmark.PINKY_TIP]
                wrist = handLms.landmark[mp_hands.HandLandmark.WRIST]
                pt1 = [int(index_tip.x * width), int(index_tip.y * height)]
                pt2 = [int(middle_tip.x * width), int(middle_tip.y * height)]
                pt3 = [int(ring_tip.x * width), int(ring_tip.y * height)]
                pt4 = [int(pinky_tip.x * width), int(pinky_tip.y * height)]
                pt0=[int(wrist.x * width), int(wrist.y * height)]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.line(frame, pt3, pt4, (0, 255, 0), 2)
                dist1=((index_tip.x-wrist.x)**2+(index_tip.y-wrist.y)**2)**0.5
                dist2=((ring_tip.x-wrist.x)**2+(ring_tip.y-wrist.y)**2)**0.5
                dim = ((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2) ** 0.5
                drp = ((ring_tip.x - pinky_tip.x) ** 2 + (ring_tip.y - pinky_tip.y) ** 2) ** 0.5
                print(dist1, dist2, dim, drp)
                if dim<0.075 and drp<0.13 and dist1>0.55 and dist2>0.56:
                    cv2.imshow("vulcan", vulcan_img)
                else:
                    try:
                        cv2.destroyWindow("vulcan")
                    except:
                        pass
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

cv2.destroyAllWindows()

