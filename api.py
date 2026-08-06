from ultralytics import YOLO
import cv2
import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse, HTMLResponse
import threading

# Carregar o modelo treinado
model = YOLO(r"runs\detect\train\weights\best.pt")

app = FastAPI()

cap = None
cap_lock = threading.Lock()


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
        # inferência
        try:
            results = model(frame)
            # desenhar resultados sobre a imagem
            rendered = results[0].plot()
        except Exception:
            rendered = frame

        ret2, buffer = cv2.imencode('.jpg', rendered)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get('/', response_class=HTMLResponse)
def index():
    return "<html><body><h3>Inspeção de segurança com IA - Stream</h3><img src='/video_feed' width='720' /></body></html>"


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
