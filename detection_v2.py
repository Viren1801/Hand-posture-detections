import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.5, maxHands=2)
Posture = None
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if len(hands) == 2:

        hand1 = hands[0]
        hand2 = hands[1]
        # lmList1 = hand1["lmList"]
        # bbox1 = hand1["bbox"]
        centerPoint1 = hand1["center"]
        centerPoint2 = hand2["center"]
        handType1 = hand1["type"]
        handType2 = hand2["type"]
        # length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)
        if (handType1 == "Left") and (handType2 == "Right"):
            if centerPoint1 < centerPoint2:
                print("Armed")
                Posture = "Armed"
            else:
                print("Unarmed")
                Posture = "Unarmed"


    cv2.imshow("test", img)
    cv2.putText(img, " Status: ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (20, 255, 155), 2)
    cv2.putText(img, Posture, (480 // 3 - 150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                4.0, (20, 255, 155), 10, 10)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cap.release()
        cv2.destroyllWindows()
        break
