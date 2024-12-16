import cv2
import os
import time
import serial
import serial.tools.list_ports
import logging
from ultralytics import YOLO

# Configuración básica del logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("registro.log"),
                        logging.StreamHandler()
                    ])

# Parámetros de captura de fotogramas
output_folder = "capturas"
os.makedirs(output_folder, exist_ok=True)


def cargar_modelo_yolo(ruta_modelo, conexion_serial):
    """Carga el modelo YOLOv8 y maneja excepciones de carga."""
    try:
        modelo = YOLO(ruta_modelo)  # Cargar modelo YOLO
        logging.info("Modelo YOLO cargado exitosamente.")
        return modelo
    except Exception as e:
        logging.error(f"Error al cargar el modelo YOLO: {e}")
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


def detectar_camara():
    """Detecta automáticamente la cámara disponible."""
    for i in range(10):
        captura = cv2.VideoCapture(2)
        if captura.isOpened():
            captura.release()
            logging.info(f"Cámara detectada en índice {2}.")
            return 2
    logging.error("No se encontró una cámara disponible.")
    return None


def clasificacion_yolo(frame, modelo):
    """
    Clasifica un fotograma utilizando el modelo YOLOv8.
    """
    try:
        # Realizar la predicción
        resultados = modelo.predict(frame, imgsz=640, conf=0.5, verbose=False)  # imgsz y conf ajustables
        detecciones = resultados[0].boxes  # Obtener las detecciones

        # Si hay detecciones, retornar información relevante
        if len(detecciones) > 0:
            for det in detecciones:
                clase = int(det.cls)  # Clase detectada (ajusta según tu entrenamiento)
                confianza = float(det.conf)  # Confianza de la detección
                logging.info(f"Detección: Clase {clase}, Confianza {confianza:.2f}")
                # Clasifica como buena (> 0.5) o mala (<= 0.5)
                print(confianza)
                return confianza > 0.5  # Devuelve True si es buena, False si es mala
        else:
            return None  # No hay detección

    except Exception as e:
        logging.error(f"Error en clasificación con YOLOv8: {e}")
        return None


def procesar_frame(captura_imagen, modelo, conexion_serial, activo):
    """Procesa cada frame capturado y clasifica si detecta algo relevante."""
    ret, frame = captura_imagen.read()
    if not ret:
        logging.error("Error al capturar la imagen.")
        conexion_serial.write(b'E')
        return activo

    # Clasificar usando YOLO
    resultado_clasificacion = clasificacion_yolo(frame, modelo)

    if resultado_clasificacion is not None:
        # Guardar el fotograma con detección
        timestamp = int(time.time())
        filepath = os.path.join(output_folder, f"deteccion_{timestamp}.jpg")
        cv2.imwrite(filepath, frame)
        logging.info(f"Fotograma con detección guardado: {filepath}")

        # Enviar señal al Arduino según el estado
        if resultado_clasificacion:
            logging.info("Criolla clasificada como en buen estado.")
            conexion_serial.write(b'D')  # Señal de buena calidad
            print('D')
        else:
            logging.info("Criolla clasificada como en mal estado.")
            conexion_serial.write(b'F')  # Señal de mala calidad
            print('F')
        time.sleep(0.5)  # Pequeña pausa para evitar envíos repetitivos

    # Mostrar el frame en tiempo real
    cv2.imshow('Principal', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Salir con la tecla 'q'
        activo = False

    return activo


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

    modelo = cargar_modelo_yolo('best3.pt', conexion_serial)
    if modelo is None:
        return

    estado = {'activo': False}

    while True:
        if conexion_serial.in_waiting > 0:
            senal = conexion_serial.read().decode('utf-8', errors='ignore').strip()
            captura_imagen = manejar_senal(senal, estado, captura_imagen, conexion_serial)

        if captura_imagen is not None and estado['activo']:
            estado['activo'] = procesar_frame(captura_imagen, modelo, conexion_serial, estado['activo'])

if __name__ == "__main__":
    main()
