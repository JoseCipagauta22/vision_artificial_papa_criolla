import cv2
import numpy as np
import tensorflow as tf
import serial
import serial.tools.list_ports
import logging
import os
import time

# Configuración básica del logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("registro.log"),
                        logging.StreamHandler()
                    ])

# Parámetros de captura de fotogramas
output_folder = "capturas"
capture_interval = 3  # Intervalo entre cada conjunto de capturas en segundos
frames_per_capture = 7  # Número de fotogramas capturados en cada ciclo
frame_delay_ms = 1  # Intervalo en milisegundos entre cada fotograma capturado en el ciclo

# Variables para control de tiempo
ultima_captura = 0

# Crear el directorio para almacenar fotogramas
os.makedirs(output_folder, exist_ok=True)


def cargar_modelo(ruta_modelo, conexion_serial):
    """Carga el modelo de TensorFlow y maneja excepciones de carga."""
    try:
        modelo = tf.keras.models.load_model(ruta_modelo)
        logging.info("Modelo cargado exitosamente.")
        return modelo
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        if conexion_serial:
            conexion_serial.write(b'C')  # Señal de fallo en la carga
        return None


def configurar_serial(puerto, baudios):
    """Configura la conexión serial con el dispositivo."""
    try:
        conexion = serial.Serial(puerto, baudios, timeout=1)
        logging.info(f"Conexión serial establecida en {puerto}.")
        return conexion
    except Exception as e:
        logging.error(f"Error al establecer conexión serial en {puerto}: {e}")
        return None


def detectar_puerto_serial():
    """Detecta automáticamente el puerto serial disponible para Arduino."""
    puertos = list(serial.tools.list_ports.comports())
    for puerto in puertos:
        try:
            conexion = serial.Serial(puerto.device, 9600, timeout=1)
            conexion.close()
            logging.info(f"Puerto serial detectado: {puerto.device}")
            return puerto.device
        except (OSError, serial.SerialException):
            logging.warning(f"No se pudo conectar al puerto {puerto.device}.")
    logging.error("No se encontró un puerto serial disponible.")
    return None


def detectar_camara(): #PELIGRO, PENDIENTE DE EDITAR#
    """Detecta automáticamente la cámara disponible."""
    for i in range(10):
        captura = cv2.VideoCapture(2)
        if captura.isOpened():
            captura.release()
            logging.info(f"Cámara detectada en índice {2}.")
            return 2
    logging.error("No se encontró una cámara disponible.")
    return None


def clasificacion_imagen(imagen, modelo):
    """Clasifica una imagen utilizando el modelo proporcionado."""
    try:
        imagen = cv2.resize(imagen, (224, 224))
        imagen = np.expand_dims(imagen, axis=0) / 255.0
        prediccion = modelo.predict(imagen)
        return prediccion[0][0]
    except Exception as e:
        logging.error(f"Error en clasificación de imagen: {e}")
        return None


def detectar_amarillo(frame):
    """Detecta color amarillo en el frame y retorna el rectángulo que lo contiene."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    amarillo_bajo = np.array([20, 100, 100])
    amarillo_alto = np.array([30, 255, 255])
    mascara = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        contorno = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contorno)
        return True, (x, y, w, h)
    else:
        return False, None


def manejar_senal(senal, estado, captura_imagen, conexion_serial):
    """Maneja las señales recibidas desde Arduino."""
    try:
        if senal == 'V':  # Arrancar
            logging.info("Arranque recibido. Iniciando procesamiento.")
            estado['activo'] = True
            if captura_imagen is None or not captura_imagen.isOpened():
                indice_camara = detectar_camara()
                captura_imagen = cv2.VideoCapture(indice_camara)
        elif senal == 'R':  # Parar
            logging.info("Parar recibido. Deteniendo procesamiento.")
            estado['activo'] = False
        elif senal == 'S':  # Parada de emergencia
            logging.info("Parada de emergencia recibida. Deteniendo de inmediato.")
            estado['activo'] = False
            if captura_imagen is not None and captura_imagen.isOpened():
                captura_imagen.release()
            cv2.destroyAllWindows()
        elif senal == 'Z':  # Sensor de presencia
            logging.info("Sensor de presencia activado (señal Z)")
        return captura_imagen
    except Exception as e:
        logging.error(f"Error manejando señal {senal}: {e}")
        return None

import time

def procesar_frame(captura_imagen, modelo, conexion_serial, activo, contador_buenas, contador_malas):
    """Procesa cada frame capturado, detecta amarillo y clasifica la imagen."""
    global ultima_captura
    umbral_malo = 0.25  # El porcentaje máximo de fotogramas "malos" permitidos para considerar una criolla "buena"
    tiempo_espera_envio = 0.5  # Pausa en segundos antes de enviar señal después de procesar una criolla

    ret, frame = captura_imagen.read()
    if not ret:
        logging.error("Error al capturar la imagen.")
        conexion_serial.write(b'E')
        return activo, contador_buenas, contador_malas

    detectado, rect = detectar_amarillo(frame)
    if detectado and (time.time() - ultima_captura >= capture_interval):
        ultima_captura = time.time()
        carpeta_actual = os.path.join(output_folder, f"deteccion_{int(time.time())}")
        os.makedirs(carpeta_actual, exist_ok=True)

        # Contadores para fotogramas en buen y mal estado
        contador_buenas = 0
        contador_malas = 0

        # Captura y clasificación de fotogramas
        for i in range(frames_per_capture):
            ret, frame_capture = captura_imagen.read()
            if not ret:
                logging.error("Error al capturar el fotograma.")
                break
            filename = os.path.join(carpeta_actual, f"capture_{i}.jpg")
            cv2.imwrite(filename, frame_capture)
            logging.info(f"Fotograma guardado como {filename}")

            # Clasificación de la imagen
            if modelo is not None:
                valor_prediccion = clasificacion_imagen(frame_capture, modelo)
                if valor_prediccion is not None:
                    if valor_prediccion > 0.5:
                        contador_buenas += 1
                    else:
                        contador_malas += 1

            # Espera entre capturas de fotogramas
            time.sleep(frame_delay_ms / 1000.0)

        # Cálculo del porcentaje de fotogramas "malos"
        total_frames = contador_buenas + contador_malas
        if total_frames > 0:
            porcentaje_malos = contador_malas / total_frames

            # Enviar señal según el porcentaje de fotogramas malos
            if porcentaje_malos > umbral_malo:
                logging.info("Criolla clasificada como en mal estado.")
                conexion_serial.write(b'F')  # Señal de mal estado
                print(1)
            else:
                logging.info("Criolla clasificada como en buen estado.")
                conexion_serial.write(b'D')  # Señal de buen estado
                print(0)
            # Espera para evitar enviar señal para la próxima criolla inmediatamente
            time.sleep(tiempo_espera_envio)

        # Dibujar el rectángulo en el frame y mostrar el estado
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    else:
        # Cierra la ventana si no se detecta amarillo
        if cv2.getWindowProperty('Amarillo', cv2.WND_PROP_VISIBLE) >= 0:
            print('Procesando')
        cv2.putText(frame, 'No se encuentra la papa criolla', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return activo, contador_buenas, contador_malas


def main():
    puerto_serial = detectar_puerto_serial()
    if puerto_serial is None:
        logging.error("No se pudo establecer conexión con el Arduino.")
        return

    conexion_serial = configurar_serial(puerto_serial, 9600)
    if conexion_serial is None:
        logging.error("Error en la configuración serial.")
        return

    conexion_serial.write(b'A')

    indice_camara = detectar_camara()
    if indice_camara is None:
        logging.error("No se pudo abrir la cámara.")
        conexion_serial.write(b'B')
        return

    captura_imagen = cv2.VideoCapture(indice_camara)
    if not captura_imagen.isOpened():
        logging.error("No se pudo abrir la cámara.")
        conexion_serial.write(b'B')
        return

    modelo = cargar_modelo('potato_classifier_model.h5', conexion_serial)
    if modelo is None:
        return

    estado = {'activo': False}
    contador_buenas = 0
    contador_malas = 0

    while True:
        if conexion_serial.in_waiting > 0:
            senal = conexion_serial.read().decode('utf-8', errors='ignore').strip()
            captura_imagen = manejar_senal(senal, estado, captura_imagen, conexion_serial)

        if captura_imagen is not None and estado['activo']:
            estado['activo'], contador_buenas, contador_malas = procesar_frame(
                captura_imagen, modelo, conexion_serial, estado['activo'], contador_buenas, contador_malas
            )

            cv2.imshow('Principal', captura_imagen.read()[1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if captura_imagen is not None:
        captura_imagen.release()
    conexion_serial.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
