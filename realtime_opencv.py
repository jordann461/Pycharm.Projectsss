#Bu kod, bir kişinin kameradan çekilen görüntüdeki yüzü referans resimleriyle
#karşılaştırarak tanıyabilen bir yüz tanıma uygulaması oluşturur. Başarıyla tanınan bir
#yüz varsa ekranda "Merhaba Kubra" mesajını gösterir; aksi takdirde "Eşleşme bulunamadı" mesajını gösterir.
import cv2
import numpy as np
import os
import time
import psutil

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_images_from_folder(folder, id):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                roi = img[y:y + h, x:x + w]
                images.append(roi)
                labels.append(id)
    return images, labels

folder_path = r'C:\temp\kubset'
images, ids = load_images_from_folder(folder_path, 1)  # 1, kişi
if len(images) == 0:
    print("Yeterli eğitim verisi bulunamadı.")
else:
    recognizer.train(images, np.array(ids))
    recognizer.save('trainer.yml')
    print("Model eğitildi ve kaydedildi.")

recognizer.read('trainer.yml')
print("Model yüklendi.")

cap = cv2.VideoCapture(0)


cpu_usages = []
ram_usages = []
start_time = time.time()

while True:
    frame_start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(roi_gray)

        threshold = 50 #(Confidence Threshold)
        if confidence < threshold:
            cv2.putText(frame, "Merhaba Kubra", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Eslesme bulunamadi", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

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

    cv2.imshow('Yüz Tanıma', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
