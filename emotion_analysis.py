"""Facial Emotion Recognition (FER), yüz ifadelerinden insan duygularını tanımlama ve kategorize etme sürecidir.
Burada; DeepFace(insan yüzlerini tanımak için kullanılan derin öğrenme tabanlı bir yapay zeka modeli)
kullanarak yüz ifadelerinden duygu tanıma işlemi yaptım.İlk önce FER-2013 veri setindeki test klasöründen
görüntüler seçip yükledim.Sonra görüntüleri DeepFace ile işledim. Deepface görüntüdeki kişinin
 baskın olan duygusunu belirledi.
 En son optimizasyon için performans ölçümü yaptım """

import cv2
import time
import psutil
from deepface import DeepFace
import matplotlib.pyplot as plt
import tensorflow as tf

# GPU kullanmamak için ayar
tf.config.set_visible_devices([], 'GPU')

# Yüz ve göz tespiti için haarcascade dosyaları
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Görüntü dosyasının yolu
img_path = r'C:\temp\toplu.jpg'

# Görüntüyü yükleme
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Resim yüklenemedi: {img_path}")

# Görüntüyü griye dönüştürme
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("Yüz tespit edilemedi.")
else:
    print(f"Tespit edilen yüz sayısı: {len(faces)}")

# Performans ölçümleri için zaman ve sistem kaynakları takibi
start_time = time.time()
start_memory = psutil.virtual_memory().percent
start_cpu = psutil.cpu_percent(interval=None)

# Duygu analizi
try:
    result = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
    detected_emotion = result[0]['dominant_emotion']
except Exception as e:
    detected_emotion = "Yüz tespit edilemedi"
    print(f"Hata: {e}")

end_time = time.time()
end_memory = psutil.virtual_memory().percent
end_cpu = psutil.cpu_percent(interval=None)

# Sonuçlar
print("Tespit edilen duygu:", detected_emotion)
print(f"Yanıt süresi: {end_time - start_time:.2f} saniye")
print(f"RAM Kullanımı: {end_memory - start_memory:.2f}%")
print(f"CPU Kullanımı: {end_cpu - start_cpu:.2f}%")

# Görüntüyü gösterme
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
