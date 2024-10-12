from deepface import DeepFace


img_path = r'C:\temp\fer2013\test\happy\PrivateTest_647018.jpg'

result = DeepFace.analyze(img_path, actions=['emotion'])

emotions_detected = result[0]['emotion'].keys()

print("Tespit edilen duygu türleri:", list(emotions_detected))


img_path = r'C:\temp\fer2013\test\happy\PrivateTest_647018.jpg'

models = ["VGG-Face", "Facenet", "OpenFace", "DeepID", "ArcFace"]

for model in models:
    try:
        model_instance = DeepFace.build_model(model_name=model)
        print(f"Model: {model} - Destekleniyor")

        result = DeepFace.analyze(img_path, actions=['emotion'])
        dominant_emotion = result[0]['dominant_emotion']
        print(f"Model: {model}, Dominant Emotion: {dominant_emotion}")

    except Exception as e:
        print(f"Model: {model} - Hata: {str(e)}")
