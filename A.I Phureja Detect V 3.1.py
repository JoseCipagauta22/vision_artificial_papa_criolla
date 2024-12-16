from ultralytics import YOLO
import cv2
import serial
import serial.tools.list_ports
import logging
import time
from ultralytics.utils import LOGGER

from jinja2 import Template

# Configurar el logger de YOLO para silenciar mensajes
LOGGER.setLevel("CRITICAL")

# Configuración básica del logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("registro.log"),
                        logging.StreamHandler()
                    ])


template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ titulo }}</title>
</head>
<body>
    <h1>{{ encabezado }}</h1>
    <p>{{ contenido }}</p>
</body>
</html>
""")

html = template.render(
    titulo="Página Dinámica",
    encabezado="Bienvenido a Jinja2",
    contenido="Este contenido fue generado dinámicamente con Python."
)

# Guardar el HTML
with open("dinamico.html", "w") as file:
    file.write(html)


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


def manejar_senal(senal, estado, captura_imagen, conexion_serial, model_path):
    """Maneja las señales recibidas desde Arduino para controlar el flulpjo del programa."""
    try:
        if senal == 'V':  # Arrancar
            logging.info("Señal de arranque recibida. Iniciando procesamiento.")
            estado['activo'] = True
            if not captura_imagen or not captura_imagen.isOpened():
                captura_imagen = cv2.VideoCapture(2)
                if not captura_imagen.isOpened():
                    logging.error("No se pudo abrir la cámara. Enviando señal 'B'.")
                    conexion_serial.write(b'B')  # Enviar señal de error en la cámara
            if not estado['modelo']:
                estado['modelo'] = YOLO(model_path)
                conexion_serial.write(b'A')  # Confirmar arranque

        elif senal == 'R':  # Detener
            logging.info("Señal de parada recibida. Deteniendo procesamiento temporal.")
            estado['activo'] = False

        elif senal == 'S':  # Parada de emergencia
            logging.info("Señal de parada de emergencia recibida. Deteniendo todo.")
            estado['activo'] = False
            if captura_imagen and captura_imagen.isOpened():
                captura_imagen.release()
            cv2.destroyAllWindows()
            estado['modelo'] = None

        return captura_imagen
    except Exception as e:
        logging.error(f"Error manejando señal {senal}: {e}")
        return None


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

    # Enviar señal de inicialización
    conexion_serial.write(b'A')

    # Intentar cargar el modelo
    try:
        modelo = YOLO(model_path)
        logging.info("Modelo cargado exitosamente.")
        conexion_serial.write(b'A')  # Confirmar éxito
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        conexion_serial.write(b'C')  # Señal de error en la carga del modelo
        return

    # Variables de estado
    estado = {'activo': False, 'modelo': modelo}
    captura_imagen = None
    criolla_presente = False  # Bandera para detectar cambio de estado
    criolla_estado = None  # Estado actual de la criolla (buena o mala)
    last_time = time.time()  # Tiempo inicial para cálculo de FPS

    while True:
        try:
            # Leer señal del Arduino
            if conexion_serial.in_waiting > 0:
                senal = conexion_serial.read().decode('utf-8', errors='ignore').strip()
                captura_imagen = manejar_senal(senal, estado, captura_imagen, conexion_serial, model_path)

            # Procesar si el estado está activo
            if estado['activo'] and captura_imagen:
                ret, frame = captura_imagen.read()
                if not ret:
                    logging.error("Error al capturar el frame de la cámara. Enviando señal 'E'.")
                    conexion_serial.write(b'E')  # Señal de error en la captura del frame
                    continue

                # Calcular FPS
                current_time = time.time()
                fps = 1 / (current_time - last_time)
                last_time = current_time
                #logging.info(f"FPS: {fps:.2f}")

                # Hacer predicciones con YOLOv8
                results = estado['modelo'](frame, conf=conf_threshold)

                # Detectar si hay criolla presente
                detecciones = [(result.cls.item(), result.conf.item()) for result in results[0].boxes]
                if detecciones:
                    logging.info(f"Detección: {detecciones}")
                    criolla_estado = "F" if any(cls == 0 for cls, _ in detecciones) else "D"  # Clase 0 = buena
                    criolla_presente = True  # Marcar criolla como presente
                else:
                    # Si no hay detección y antes había una criolla presente, enviar la señal correspondiente
                    if criolla_presente:
                        logging.info(f"Criolla clasificada como {'buena' if criolla_estado == 'D' else 'mala'}.")
                        conexion_serial.write(criolla_estado.encode())  # Enviar señal de estadoe
                        print(criolla_estado)
                        criolla_presente = False  # Reiniciar estado para la próxima detección
                        criolla_estado = None  # Reiniciar estado de criolla

                # Dibujar las predicciones y mostrar los FPS en la ventana
                annotated_frame = results[0].plot()
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Detecciones en vivo - Cámara", annotated_frame)

                # Salir si se presiona 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Error inesperado en el bucle principal: {e}")
            conexion_serial.write(b'E')  # Señal de error general al Arduino
            break

    # Liberar recursos al finalizar
    if captura_imagen and captura_imagen.isOpened():
        captura_imagen.release()
        print('test')
    conexion_serial.close()
    cv2.destroyAllWindows()


# Configuración
if __name__ == "__main__":
    # Ruta al modelo YOLO entrenado
    model_path = "best3.pt"

    # Ejecutar detección con Arduino
    detectar_yolo_arduino(model_path, conf_threshold=0.1)
