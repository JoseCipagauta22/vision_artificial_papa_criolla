import cv2
import numpy as np
import tensorflow as tf
#import serial
import time

# Cargar el modelo
model = tf.keras.models.load_model('potato_classifier_model.h5')

# Configurar la conexión serial con Arduino
#ser = serial.Serial('/dev/ttyACM0', 9600)  # Asegúrate de usar el puerto correcto

# Función para predecir la clase de una imagen
def classify_image(image):
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return prediction[0][0]

# Capturar y clasificar imágenes desde la cámara USB
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pred = classify_image(frame)
    if pred < 0.5:
        print("Criolla en buen estado detectada.")
        #ser.write(b'1')  # Enviar señal a Arduino
        time.sleep(1)    # Evitar múltiples señales seguidas
    else:
        print("Criolla en mal estado detectada.")
        #ser.write(b'0')  # Enviar señal a Arduino

    time.sleep(2)  # Intervalo entre capturas para evitar sobrecargar el sistema

cap.release()
#ser.close()
