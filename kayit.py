#Kodun amacı, gerçek zamanlı yüz tanıma işlemini gerçekleştirmek ve
# tanınmayan yüzler için kullanıcıdan kayıt alarak veritabanını güncellemektir.
# İlk olarak, Excel dosyasından yüz verileri yüklenir ve
# bu verilerle bir yüz tanıma modeli eğitilir.
# Kamera akışı başlatılır ve her görüntü karesi yüz tespiti için taranır;
# tanınan yüzler ekranda kişinin bilgileriyle birlikte gösterilir.
# Eğer yüz tanınamazsa, kullanıcıdan yeni kişi bilgileri alınır,
# mevcut yüz fotoğrafı kaydedilir ve bu bilgiler Excel dosyasına eklenir.

import cv2
import numpy as np
import pandas as pd
import os
import time


def load_face_data(file_path, image_size=(100, 100)):

    df = pd.read_excel(file_path)
    face_images = []
    labels = []

    for index, row in df.iterrows():
        image_path = row['Fotoğraf Yolu']
        if not os.path.isfile(image_path):
            print(f"Dosya bulunamadı veya geçersiz yol: {image_path}")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Resim yüklenirken hata: {image_path}")
            continue

        image = cv2.resize(image, image_size)
        face_images.append(image)
        labels.append(index)

    print(f"{len(face_images)} resim ve {len(labels)} etiket yüklendi.")
    return face_images, labels, df


def save_face_image(frame, x, y, w, h, save_path):

    face_region = frame[y:y + h, x:x + w]
    file_name = f"face_{int(time.time())}.jpg"
    full_path = os.path.join(save_path, file_name)
    cv2.imwrite(full_path, face_region)
    return full_path


def add_new_record(file_path, photo_path, name, surname, age, gender):

    df = pd.read_excel(file_path)


    new_record = pd.DataFrame({
        'Fotoğraf Yolu': [photo_path],
        'Ad': [name],
        'Soyad': [surname],
        'Yaş': [age],
        'Cinsiyet': [gender]
    })


    df = pd.concat([df, new_record], ignore_index=True)


    df.to_excel(file_path, index=False)
    print("Yeni kayıt eklendi.")


def main():

    file_path = r'C:\temp\veritabani.xlsx'
    face_images, labels, df = load_face_data(file_path)

    if len(face_images) == 0 or len(labels) == 0:
        print("Eğitim için kullanılacak yüz resmi veya etiket bulunmuyor.")
        return


    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_images, np.array(labels))
    print("Eğitim tamamlandı.")


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)


    save_folder = 'biometrics'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]
            face_region = cv2.resize(face_region, (100, 100))
            label, confidence = recognizer.predict(face_region)

            if confidence < 100:
                person_info = df.iloc[label]
                name = f"Ad: {person_info.get('Ad', 'Bilgi Yok')}"
                surname = f"Soyad: {person_info.get('Soyad', 'Bilgi Yok')}"
                age = f"Yaş: {person_info.get('Yaş', 'Bilgi Yok')}"
                gender = f"Cinsiyet: {person_info.get('Cinsiyet', 'Bilgi Yok')}"
            else:
                name = "Kişi Bulunamadı"
                surname = ""
                age = ""
                gender = ""

                print("Kişi bulunamadı. Yeni kayıt eklemek için bilgileri girin.")
                photo_path = save_face_image(frame, x, y, w, h, save_folder)
                name = input("Ad: ")
                surname = input("Soyad: ")
                age = input("Yaş: ")
                gender = input("Cinsiyet: ")

                add_new_record(file_path, photo_path, name, surname, age, gender)


            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


            y_offset = y - 10
            cv2.putText(frame, name, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            y_offset += 30
            cv2.putText(frame, surname, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            y_offset += 30
            cv2.putText(frame, age, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            y_offset += 30
            cv2.putText(frame, gender, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
