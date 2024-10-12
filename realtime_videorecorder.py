#gerçek zamanlı video kayıt uygulaması ve duygu analizi

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap.set(cv2.CAP_PROP_FPS, 60)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)


        if isinstance(result, list):
            result = result[0]


        dominant_emotion = result['dominant_emotion']
        cv2.putText(frame, f'Duygu: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print(f"Error during emotion analysis: {e}")


    cv2.imshow('Yüksek FPS, Düşük Çözünürlük ve Duygu Analizi', frame)


    out.write(frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
