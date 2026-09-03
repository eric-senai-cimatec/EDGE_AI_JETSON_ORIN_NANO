from ultralytics import YOLO
import cv2
import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse, HTMLResponse
import threading
import Jetson.GPIO as GPIO

RELAY_PIN = 7  # GPIO pin number

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)

# Carregar o novo modelo TensorRT bruto
model = YOLO('runs/detect/train/weights/best.engine')

app = FastAPI()
cap = None
cap_lock = threading.Lock()

def turn_on_gpio():
    """Turn on the relay."""
    GPIO.output(RELAY_PIN, GPIO.HIGH)

def turn_off_gpio():
    """Turn off the relay."""
    GPIO.output(RELAY_PIN, GPIO.LOW)

def get_capture():
    """Get the video capture object, ensuring thread safety."""
    global cap
    with cap_lock:
        if cap is None:
            cap = cv2.VideoCapture(0)
        return cap

def gen_frames():
    """Generate frames from the camera and process them with the YOLO model."""
    cap = get_capture()
    last_state = False  # Evita o spam de comandos elétricos no pino GPIO
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # 2. Força a resolução idêntica à exportação e define a GPU explicitamente
            results = model(frame, imgsz=640, conf=0.25, device=0)
            
            helmet_detected = False

            # 3. Varredura mantendo o seu formato padrão (1, 15, 8400)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    print(f'Classe detectada: {cls_name}')
                    
                    if cls_name == 'no_helmet':
                        helmet_detected = True

            # 4. Só altera o estado elétrico se houver uma MUDANÇA real no quadro
            if helmet_detected != last_state:
                if helmet_detected:
                    turn_on_gpio()
                else:
                    turn_off_gpio()
                last_state = helmet_detected  # Atualiza o histórico
                
            # Desenha os resultados na tela
            rendered = results[0].plot()
            
        except Exception as e:
            print(f"Erro no loop de processamento: {e}")
            rendered = frame
            
        ret2, buffer = cv2.imencode('.jpg', rendered)
        if not ret2:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get('/', response_class=HTMLResponse)
def index():
    """Serve the main page with the video feed."""
    return """<html><body><h3>Inspeção de segurança com IA - Stream</h3><img src='/video_feed' width='1280' /></body></html>"""

@app.get('/video_feed')
def video_feed():
    """Serve the video feed as a multipart response."""
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        uvicorn.run(app, host='127.0.0.1', port=8000)
    finally:
        # Garante a limpeza segura dos pinos ao fechar a aplicação
        GPIO.cleanup()
