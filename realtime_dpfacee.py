import cv2
from deepface import DeepFace
import os

def load_reference_images_from_folder(folder_path):
    reference_images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            reference_images.append(img)
    return reference_images

def resize_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def recognize_face(frame, reference_images):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for ref_img in reference_images:
        try:
            result = DeepFace.verify(frame_rgb, ref_img, model_name='VGG-Face', enforce_detection=False)
            if result['verified']:
                return True
        except Exception as e:
            print(f"Error: {e}")
            continue
    return False

def main():

    folder_path = r'C:\temp\kubset'
    reference_images = load_reference_images_from_folder(folder_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılmadı.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Kamera görüntüsü alınamadı.")
            break

        frame = resize_frame(frame)


        face_detected = recognize_face(frame, reference_images)
        if face_detected:
            cv2.putText(frame, 'Merhaba Kubra', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Eslesme bulunamadı', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
