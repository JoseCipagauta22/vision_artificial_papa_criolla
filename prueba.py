from ultralytics import YOLO
import cv2
import serial
import serial.tools.list_ports
import logging
import time
from ultralytics.utils import LOGGER

# Configurar el logger de YOLO para silenciar mensajes
LOGGER.setLevel("CRITICAL")

# Configuración básica del logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("registro.log"),
                        logging.StreamHandler()
                    ])


# Función para procesar un cuadro con el modelo YOLO
def process_frame(frame, model):
    """Procesa un cuadro para detectar objetos usando YOLO."""
    frame_resized = cv2.resize(frame, dsize=(640, 420))  # Redimensionar para consistencia
    results = model(frame_resized, conf=0.3)  # Detectar con un umbral de confianza

    detection_counter = {}  # Contador para las detecciones
    for result in results:
        for box in result.boxes:
            cls = box.cls[0].item()  # Índice de clase
            name = model.names[int(cls)]  # Nombre de la clase

            if name != 'hojaSana':  # Ignorar la clase específica
                if name not in detection_counter:
                    detection_counter[name] = 1
                else:
                    detection_counter[name] += 1
                return frame_resized, name, box.xyxy[0], box.conf[0].item()

    return frame_resized, None, None, None  # Si no se detecta nada relevante


# Configuración de la conexión serial
def configurar_serial(puerto, baudios=9600):
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


# Función principal de detección
def detectar_yolo_arduino(model_path, conf_threshold=0.1):
    """Detecta objetos en tiempo real utilizando YOLOv8, controlado por señales de Arduino."""
    # Configurar la conexión serial
    puerto_serial = detectar_puerto_serial()
    if puerto_serial is None:
        logging.error("No se pudo establecer conexión con el Arduino.")
        return
    conexion_serial = configurar_serial(puerto_serial)
    if conexion_serial is None:
        logging.error("Error en la configuración serial.")
        return

    # Variables de estado
    captura_imagen = cv2.VideoCapture(2)
    if not captura_imagen.isOpened():
        logging.error("No se pudo abrir la cámara. Terminando programa.")
        return

    try:
        modelo = YOLO(model_path)
        logging.info("Modelo cargado exitosamente.")
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        return

    frame_count = 0
    skip_frames = 2

    while True:
        try:
            ret, frame = captura_imagen.read()
            if not ret:
                logging.error("Error al capturar el frame de la cámara.")
                break

            # Saltar frames para optimización
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Procesar el frame con YOLO
            processed_frame, name, bbox, conf = process_frame(frame, modelo)

            # Si se detecta un objeto
            if name:
                logging.info(f"Detección: {name} con confianza {conf:.2f}")
                # Enviar señal específica al Arduino
                if name == "criolla":
                    conexion_serial.write(b'D')  # Señal para criolla buena
                else:
                    conexion_serial.write(b'F')  # Señal para criolla mala

            # Mostrar el frame procesado
            cv2.imshow("Detección YOLO", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")
            break


    # Liberar recursos
    captura_imagen.release()
    conexion_serial.close()
    cv2.destroyAllWindows()


# Configuración principal
if __name__ == "__main__":
    model_path = "best3.pt"  # Ruta al modelo YOLO
    detectar_yolo_arduino(model_path, conf_threshold=0.3)
