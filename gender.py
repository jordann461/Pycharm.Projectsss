import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

img_path = r'C:\temp\safis.jpg'


img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Resim yüklenemedi: {img_path}")


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()


resp = DeepFace.analyze(img_path)
print(resp)
