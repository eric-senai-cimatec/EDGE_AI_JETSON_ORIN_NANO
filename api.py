from ultralytics import YOLO
import cv2
import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse, HTMLResponse
import threading
# import Jetson.GPIO as GPIO

# RELAY_PIN = 7  # GPIO pin number for the relay control (BOARD numbering)

# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BOARD)

# Configura o pino apenas uma vez
# GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)

# Carregar o modelo treinado
model = YOLO('runs/detect/train/weights/best.pt')

app = FastAPI()
cap = None
cap_lock = threading.Lock()


# def turn_on_gpio():
#     """Turn on the relay."""
#     GPIO.output(RELAY_PIN, GPIO.HIGH)


# def turn_off_gpio():
#     """Turn off the relay."""
#     GPIO.output(RELAY_PIN, GPIO.LOW)


def get_capture():
    global cap
    with cap_lock:
        if cap is None:
            cap = cv2.VideoCapture(0)
        return cap


def gen_frames():
    cap = get_capture()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model(frame)

            # Variável para controlar se um capacete foi achado neste quadro
            helmet_detected = False

            # Imprimir classes detectadas no terminal ---
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    print(f'Classe detectada: {cls_name}')

                    if cls_name == 'helmet':
                        helmet_detected = True

            # Aciona o GPIO apenas uma vez por quadro, baseado no resultado geral
            # if helmet_detected:
            #     turn_on_gpio()
            # else:
            #     turn_off_gpio()

            # desenhar resultados sobre a imagem
            rendered = results[0].plot()
        except Exception as e:
            print(e)
            rendered = frame

        ret2, buffer = cv2.imencode('.jpg', rendered)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get('/', response_class=HTMLResponse)
def index():
    return """<html><body><h3>Inspeção de segurança com IA - Stream</h3><img src='/video_feed' width='1280' /></body></html>"""


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        uvicorn.run(app, host='127.0.0.1', port=8000)
    finally:
        pass
        # Limpa os pinos GPIO ao fechar o programa
        # GPIO.cleanup()
