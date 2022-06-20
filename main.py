import cv2
import math
import numpy as np
import mediapipe as mp
import threading
import time

global wrist_x_left
global wrist_x_right
Posture = None

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    _, image = cap.read()
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    point = mp_hands.HandLandmark.WRIST
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
        if results.multi_hand_landmarks:
            image_Height, image_Width, _ = image.shape
            annotated_image = cv2.flip(image.copy(), 1)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                # for point in mp_hands.HandLandmark[mp_hands.HandLandmark.WRIST]:
                #     print(point)
                normalizedLandmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                       normalizedLandmark.y,
                                                                                       image_Width, image_Height)

                # print(point)
                # print(pixelCoordinatesLandmark)
                # print(normalizedLandmark)
            image = cv2.flip(annotated_image, 1)


        def get_left_hand():
            global wrist_x_left
            if results.multi_handedness:
                for idx, classification in enumerate(results.multi_handedness):
                    label = classification.classification[0].label
                    hand_landmarks_left = results.multi_hand_landmarks[idx]
                    if label == "Left":
                        wrist_x_left = hand_landmarks_left.landmark[mp_hands.HandLandmark.WRIST].x
            return wrist_x_left


        def get_right_hand():
            global wrist_x_right
            if results.multi_handedness:
                for idx, classification in enumerate(results.multi_handedness):
                    label = classification.classification[0].label
                    hand_landmarks_right = results.multi_hand_landmarks[idx]
                    if label == "Right":
                        wrist_x_right = hand_landmarks_right.landmark[mp_hands.HandLandmark.WRIST].x
            # print(wrist_x_right, "right")
            return wrist_x_right


        def get_posture():
            global Posture
            t1 = threading.Thread(target=get_left_hand)
            t2 = threading.Thread(target=get_right_hand)
            t1.start()
            t2.start()
            if (wrist_x_right < wrist_x_left) and (wrist_x_left - wrist_x_right > 0.0):
                Posture = "Armed"
            else:
                Posture = "Disarmed"

            return Posture

    get_posture()
    cv2.putText(image, " Status: ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (20, 255, 155), 2)
    cv2.putText(image, Posture, (480 // 3 - 150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                4.0, (20, 255, 155), 10, 10)
    cv2.imshow("sample", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cap.release()
        cv2.destroyllWindows()
        break
