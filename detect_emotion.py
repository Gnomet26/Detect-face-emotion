import cv2
import numpy as np
from tensorflow.keras.models import load_model

#======================== Modelni yuklash ================================
model = load_model('emotion_recognition_model.h5')

#=============================== Emotsiya nomlari ================================
emotsiya_nomlari = ['Jahldor', 'Yoqimsiz', 'Qayg\'u', 'Xursand', 'Normal', 'Hafa', 'Hayron']

#================================ XML faylni yuklash ============================================
yuz_kaskad_fayli = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def emotsiyani_aniqlash(yuz):
    yuz = cv2.resize(yuz, (48, 48))
    yuz = yuz.reshape(1, 48, 48, 1)
    yuz = yuz / 255.0
    prediction = model.predict(yuz)
    return np.argmax(prediction)

#============================ Qurilma kamerasiga ulanish =============================================
kadr = cv2.VideoCapture(0)

while True:
    test, frame = kadr.read()
    if not test:
        print("Kamerga ulanishda xatolik :(")
        break

    kul_rang = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    yuzlar = yuz_kaskad_fayli.detectMultiScale(kul_rang, 1.3, 5)

    for (x, y, w, h) in yuzlar:
        _gray = kul_rang[y:y+h, x:x+w]
        emotsiya = emotsiyani_aniqlash(_gray)
        cv2.putText(frame, emotsiya_nomlari[emotsiya], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (14, 201, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (14, 201, 255), 2)

    cv2.imshow('Yuz emotsiyasi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kadr.release()
cv2.destroyAllWindows()
