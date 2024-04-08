import cv2
import mediapipe as mp
import time
import pyautogui
import screen_brightness_control as sbc
import numpy as np

x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = 0

webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmark = hands[0].landmark
            for id, landmark in enumerate(landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                if id == 8:  # Index
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x1 = cx
                    y1 = cy
                if id == 4:  # Thumb
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x2 = cx
                    y2 = cy
                if id == 20:  # Pinky
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x3 = cx
                    y3 = cy
                if id == 0:  # Wrist
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x4 = cx
                    y4 = cy
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)  # Distance calculation
            dist_bright = ((x4 - x3) ** 2 + (y4 - y3) ** 2) ** (0.5)  # Distance calculation

        cv2.line(image, (x1, y1), (x2, y2), (189, 189, 189), 5)
        cv2.line(image, (x3, y3), (x4, y4), (189, 189, 189), 5)
        # print(dist_bright)
        if dist_bright > 110:
            b_level = np.interp(dist_bright - 110, [10, 70], [0, 100])  # Adjust ranges as needed
            sbc.set_brightness(int(b_level))
        elif dist_bright < 90:
            if dist > 50:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break