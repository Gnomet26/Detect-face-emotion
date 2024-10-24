import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#================================== Emotsiyalar ro'yxati ===============================
emotsiyalar = ['Jahldor', 'Yoqimsiz', 'Qayg\'u', 'Xursand', 'Normal', 'Hafa', 'Hayron']
numpy_sinfi = len(emotsiyalar)

#============================== emotsiyalar uchun bo'sh ro'yxatlar ============================
yuzlar = []
emotsiyalar_royxati = []

#=============================== suratlarni yuklash ==========================================
rasmlar_katalogi = 'F:\\Detect\\data\\train'
for i, emotsiya_ in enumerate(emotsiyalar):
    emotsiya_katalogi = os.path.join(rasmlar_katalogi, emotsiya_)
    for fayl_nomi in os.listdir(emotsiya_katalogi):
        if fayl_nomi.endswith('.jpg') or fayl_nomi.endswith('.jpeg') or fayl_nomi.endswith('.png'):
            img_path = os.path.join(emotsiya_katalogi, fayl_nomi)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resize_img = cv2.resize(img, (48, 48))
            yuzlar.append(resize_img)
            emotsiyalar_royxati.append(i) 

#====================== Listalardan numpy massivini yaratish ======================================
yuzlar = np.array(yuzlar).reshape(-1, 48, 48, 1)
emotsiyalar_royxati = np.array(emotsiyalar_royxati)

print(len(yuzlar))
print(len(emotsiyalar_royxati))

#==================================== Tasvirlarni qayta ishlash =======================================
yuzlar = yuzlar.astype('float32') / 255.0

#================================= Ma'lumotlarni trening va sinov qismiga ajratish =================================
X_train, X_test, y_train, y_test = train_test_split(yuzlar, emotsiyalar_royxati, test_size=0.2, random_state=42)

#========================================= TensorFlow modelini yaratish ==================================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(numpy_sinfi, activation='softmax')
])

#============================================ Modelni o'qitish =====================================================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

#================================================= Modelni saqlash =================================================
model.save('emotion_recognition_model.h5')