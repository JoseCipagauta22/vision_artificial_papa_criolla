import cv2
import numpy as np
import tensorflow as tf
import serial
import serial.tools.list_ports
import logging

# Configuración básica del logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("registro.log"),
                        logging.StreamHandler()
                    ])

def cargar_modelo(ruta_modelo, conexion_serial):
    """Carga el modelo de TensorFlow desde la ruta especificada y envía señal de error al Arduino si falla."""
    try:
        modelo = tf.keras.models.load_model(ruta_modelo)
        logging.info("Modelo cargado exitosamente.")
        return modelo
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        if conexion_serial:
            conexion_serial.write(b'5')  # Señal indicando fallo en la carga del modelo
        exit()

def configurar_serial(puerto, baudios):
    """Configura la conexión serial con el dispositivo."""
    try:
        conexion = serial.Serial(puerto, baudios, timeout=1)
        logging.info("Conexión serial establecida en %s.", puerto)
        return conexion
    except Exception as e:
        logging.error(f"Error al establecer la conexión serial en {puerto}: {e}")
        exit()

def detectar_puerto_serial():
    """Detecta automáticamente el puerto serial disponible para conectar con Arduino."""
    puertos = list(serial.tools.list_ports.comports())
    for puerto in puertos:
        try:
            conexion = serial.Serial(puerto.device, 9600, timeout=1)
            conexion.close()
            logging.info("Puerto serial detectado: %s", puerto.device)
            return puerto.device
        except (OSError, serial.SerialException):
            logging.warning("No se pudo conectar al puerto %s.", puerto.device)
            pass
    logging.error("No se encontró un puerto serial disponible.")
    return None

def detectar_camara():
    """Detecta automáticamente la cámara disponible y retorna el índice."""
    for i in range(10):
        captura = cv2.VideoCapture(i)
        if captura.isOpened():
            captura.release()
            logging.info("Cámara detectada en índice %d.", i)
            return i
    logging.error("No se encontró una cámara disponible.")
    return None

def clasificacion_imagenes(imagen, modelo):
    """Clasifica una imagen utilizando el modelo proporcionado."""
    imagen = cv2.resize(imagen, (150, 150))
    imagen = np.expand_dims(imagen, axis=0) / 255.0
    prediccion = modelo.predict(imagen)
    return prediccion[0][0]

def detectar_amarillo(frame):
    """Detecta el color amarillo en la imagen y retorna el rectángulo que lo contiene."""
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

def procesar_frame(frame, modelo, conexion_serial, activo, contador_buenas, contador_malas):
    """Procesa cada frame de la cámara, detecta el color amarillo y clasifica la imagen."""
    detectado, rect = detectar_amarillo(frame)

    if detectado:
        if not activo:
            logging.info("Objeto amarillo detectado.")
            activo = True
            contador_buenas = 0
            contador_malas = 0

        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        valor_prediccion = clasificacion_imagenes(frame, modelo)
        if valor_prediccion < 0.5:
            logging.info("Criolla en buen estado detectada.")
            contador_buenas += 1
            cv2.putText(frame, 'Buen estado', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            logging.info("Criolla en mal estado detectada.")
            contador_malas += 1
            cv2.putText(frame, 'Mal estado', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Amarillo', frame)

    else:
        # Verificar si la ventana 'Amarillo' existe y está visible
        if cv2.getWindowProperty('Amarillo', cv2.WND_PROP_VISIBLE) >= 0:
            try:
                cv2.destroyWindow('Amarillo')  # Intentar cerrar la ventana
            except cv2.error as e:
                logging.warning(f"No se pudo cerrar la ventana 'Amarillo': {e}")

        cv2.putText(frame, 'No se encuentra la papa criolla', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return activo, contador_buenas, contador_malas


def manejar_senal(senal, estado, captura_imagen, modelo, conexion_serial):
    """Maneja las señales recibidas desde Arduino."""
    if senal == 'V':  # Arrancar
        logging.info("Arranque recibido. Iniciando procesamiento.")
        estado['activo'] = True
        # Reabrir la cámara si ha sido cerrada
        if captura_imagen is None or not captura_imagen.isOpened():
            indice_camara = detectar_camara()
            captura_imagen = cv2.VideoCapture(indice_camara)
    elif senal == 'R':  # Parar
        logging.info("Parar recibido. Deteniendo procesamiento tras el frame actual.")
        estado['activo'] = False
    elif senal == 'S':  # Parada de emergencia
        logging.info("Parada de emergencia recibida. Deteniendo inmediatamente.")
        estado['activo'] = False
        if captura_imagen is not None and captura_imagen.isOpened():
            captura_imagen.release()
        cv2.destroyAllWindows()
    elif senal == 'Z':  # Sensor de presencia
        logging.info("Sensor de presencia activado. Señal recibida: Z")

    return captura_imagen

def main():
    # Detectar y conectar con el puerto serial
    puerto_serial = detectar_puerto_serial()
    if puerto_serial is None:
        logging.error("No se pudo establecer conexión con el Arduino.")
        return
    
    conexion_serial = configurar_serial(puerto_serial, 9600)
    
    # Enviar señal de éxito en la conexión
    conexion_serial.write(b'2')
    
    # Detectar y abrir la cámara
    indice_camara = detectar_camara()
    if indice_camara is None:
        logging.error("No se pudo abrir la cámara.")
        conexion_serial.write(b'3')
        return
    
    captura_imagen = cv2.VideoCapture(indice_camara)
    
    if not captura_imagen.isOpened():
        logging.error("No se pudo abrir la cámara.")
        conexion_serial.write(b'3')
        return
    
    # Cargar el modelo, ahora enviará una señal si falla
    modelo = cargar_modelo('potato_classifier_model.h5', conexion_serial)
    
    estado = {
        'activo': False  # Controla si el procesamiento está activo
    }

    contador_buenas = 0
    contador_malas = 0

    while True:
        # Leer señales desde el Arduino
        if conexion_serial.in_waiting > 0:
            senal = conexion_serial.read().decode('utf-8').strip()
            captura_imagen = manejar_senal(senal, estado, captura_imagen, modelo, conexion_serial)

        # Proceso de imágenes cuando el sistema está activo
        if captura_imagen is not None and estado['activo']:
            ret, frame = captura_imagen.read()
            if not ret:
                logging.error("Error al capturar la imagen.")
                conexion_serial.write(b'4')
                break

            estado['activo'], contador_buenas, contador_malas = procesar_frame(frame, modelo, conexion_serial, estado['activo'], contador_buenas, contador_malas)

            cv2.imshow('Principal', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if captura_imagen is not None:
        captura_imagen.release()
    conexion_serial.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
