#Bu kodda veritabanında kayıtlı yüzler kullanılarak bir model eğitilir.
# Bu model, kameradan alınan görüntülerdeki yüzleri tanımak için kullanılır.
# Tanınan yüzler veritabanındaki bilgilerle eşleştirilir ve ekranda isim, soyisim, yaş, cinsiyet bilgileri gösterilir.
# Eğer tanınmayan bir yüz algılanırsa, kullanıcıdan yeni bilgiler (isim, yaş, vb.) alınır ve yüz resmi kaydedilerek veritabanına eklenir.

import cv2
import os
import psycopg2
import numpy as np
import time

class FaceRecognitionDatabase:
    def __init__(self, db_name, user, password):
        self.connection = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host='localhost'
        )
        self.cursor = self.connection.cursor()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS face_records (
            image TEXT,
            name TEXT,
            surname TEXT,
            age INTEGER,
            gender TEXT
        )''')
        self.connection.commit()

    def add_record(self, image_path, name, surname, age, gender):
        self.cursor.execute('''INSERT INTO face_records (image, name, surname, age, gender)
            VALUES (%s, %s, %s, %s, %s)''', (image_path, name, surname, age, gender))
        self.connection.commit()

    def get_all_records(self):
        self.cursor.execute('SELECT * FROM face_records')
        return self.cursor.fetchall()

    def get_record_by_name(self, name):
        self.cursor.execute('SELECT * FROM face_records WHERE name = %s', (name,))
        return self.cursor.fetchone()

    def close(self):
        self.cursor.close()
        self.connection.close()

def main():
    db = FaceRecognitionDatabase(db_name='face_recognition', user='postgres', password='Jordan1945')
    db.create_table()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    video_capture = cv2.VideoCapture(0)

    records = db.get_all_records()
    names, images = [], []

    for record in records:
        image_path = record[0]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Hata: Görüntü yüklenemedi - {image_path}")
            continue

        names.append(record[1])
        images.append(img)

    if images:
        recognizer.train(images, np.array(range(len(names))))
        print("Model başarıyla eğitildi.")
    else:
        print("Eğitim için uygun veri yok.")

    info_text = "Yüz Bulunamadı"  # Başlangıçta varsayılan mesaj
    last_detection_time = time.time()
    detection_interval = 1

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if time.time() - last_detection_time >= detection_interval:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            last_detection_time = time.time()

            if len(faces) == 0:
                info_text = "Yüz Bulunamadı"  # Eğer yüz algılanmazsa bu mesaj gösterilecek
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_image = gray[y:y + h, x:x + w]

                    if names:
                        label, confidence = recognizer.predict(face_image)

                        if confidence < 70:
                            name = names[label]
                            record = db.get_record_by_name(name)
                            if record:
                                name = record[1]
                                surname = record[2]
                                age = record[3]
                                gender = record[4]
                                info_text = f"İsim: {name}, Soyisim: {surname}, Yaş: {age}, Cinsiyet: {gender}"
                                print(f"Kayıt Bulundu: {record}")
                        else:
                            info_text = "Kayıt Bulunamadı. Yeni kayıt yapılıyor."
                            print(info_text)
                            new_name = input("Yeni Kayıt - İsim: ")
                            new_surname = input("Yeni Kayıt - Soyisim: ")
                            new_age = int(input("Yeni Kayıt - Yaş: "))
                            new_gender = input("Yeni Kayıt - Cinsiyet: ")

                            if not os.path.exists('biomet'):
                                os.makedirs('biomet')

                            image_path = f'biomet/{new_name}.jpg'
                            cv2.imwrite(image_path, face_image)

                            db.add_record(image_path, new_name, new_surname, new_age, new_gender)
                            print("Kayıt Başarıyla Eklendi.")
                            return
                    else:
                        info_text = "Model eğitilmedi, yüz tespit edilemiyor."

        # Bilgileri alt alta göstermek için her bilgi satırını ayrı bir y koordinatına yazdırıyoruz
        y_offset = 50
        for i, line in enumerate(info_text.split(", ")):
            cv2.putText(frame, line, (50, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    db.close()

if __name__ == "__main__":
    main()
