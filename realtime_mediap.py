#Bu kod, gerçek zamanlı yüz algılama yapmak için Mediapipe kütüphanesini ve OpenCV kütüphanesini kullanır.
import cv2
import mediapipe as mp
import time
import psutil


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    start_time = time.time()
    cpu_usages = []
    ram_usages = []

    while True:
        frame_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        results = face_detection.process(rgb_frame)


        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)


        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time

        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().percent

        cpu_usages.append(cpu_usage)
        ram_usages.append(ram_usage)


        if time.time() - start_time >= 1:
            avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
            avg_ram_usage = sum(ram_usages) / len(ram_usages)
            print(f"Frameler arası işlem süresi: {frame_time:.4f} sn")
            print(f"Ortalama RAM kullanımı: {avg_ram_usage:.2f}%")
            print(f"Ortalama CPU kullanımı: {avg_cpu_usage:.2f}%")


            cpu_usages = []
            ram_usages = []
            start_time = time.time()

        cv2.imshow('Mediapipe Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()