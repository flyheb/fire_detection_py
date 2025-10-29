import cv2 as cv
import sys
import numpy as np
from ultralytics import YOLO

# === Configuração da câmera ===
cap = cv.VideoCapture(1)

if not cap.isOpened():
    sys.exit("Erro: não foi possível abrir o vídeo. Verifique a câmera e o índice.")

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)   # Reduzir resolução
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)            # Limitar FPS
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)      # Reduzir buffer

# === Configuração do YOLOv8 ===
# Use 'fire.pt' ou um modelo YOLOv8 especializado em fogo
# Se não tiver, teste com 'yolov8n.pt' para verificar funcionamento básico
model = YOLO("yolo_fire_mini.pt")  # substitua com o modelo correto

# === Faixas HSV para detecção de fogo (tons de laranja/vermelho) ===
lower_orange = np.array([10, 100, 180])
upper_orange = np.array([35, 255, 255])
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])
lower_white = np.array([0, 0, 220]) 
upper_white = np.array([50, 60, 255])

# Controle de brilho e flicker
last_brightness = 0

# === OTIMIZAÇÕES DE PERFORMANCE ===
# Kernel morfológico criado uma vez
kernel = np.ones((3, 3), np.uint8)

# Contador para YOLO (executa apenas a cada N frames)
yolo_counter = 0
YOLO_INTERVAL = 10  # Executa YOLO a cada 10 frames

# Frame redimensionado criado uma vez
display_size = (700, 500)

while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit("Erro: não foi possível ler o frame da câmera.")

    # === Conversões otimizadas ===
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv_blur = cv.GaussianBlur(hsv_frame, (5, 5), 0)

    # === Máscaras HSV ===
    mask_orange = cv.inRange(hsv_blur, lower_orange, upper_orange)
    mask_red1 = cv.inRange(hsv_blur, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv_blur, lower_red2, upper_red2)
    mask_white = cv.inRange(hsv_blur, lower_white, upper_white)
 
    masks = [mask_orange, mask_red1, mask_red2, mask_white]
    mask = cv.bitwise_or(masks[0], masks[1])
    for m in masks[2:]:
        mask = cv.bitwise_or(mask, m)

    result_hsv = cv.bitwise_and(frame, frame, mask=mask)

    # === OTIMIZAÇÃO: Processamento morfológico apenas quando necessário ===
    # Verificar se há pixels na máscara antes de processar
    if cv.countNonZero(mask) > 100:  # Só processa se houver pixels suficientes
        edges = cv.morphologyEx(result_hsv, cv.MORPH_CLOSE, kernel)
        edges = cv.GaussianBlur(edges, (5, 5), 0)
        
        # === Canny + Contornos ===
        edges_gray = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
        edges_gray_canny = cv.Canny(edges_gray, 60, 160)
    else:
        # Criar imagem vazia se não há pixels
        edges_gray_canny = np.zeros_like(gray_frame)

    # === Checagem de variação de brilho ===
    brightness = np.mean(gray_frame)
    flicker = abs(brightness - last_brightness)
    last_brightness = brightness

    hsv_fire_detected = False
    contours, _ = cv.findContours(edges_gray_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv.contourArea(c)
        if 100 < area and flicker > 2:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(result_hsv, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(result_hsv, "HSV Fire", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            hsv_fire_detected = True

    # === OTIMIZAÇÃO CRÍTICA: YOLO apenas a cada N frames ===
    yolo_fire_detected = False
    annotated_frame = frame
    
    yolo_counter += 1
    if yolo_counter >= YOLO_INTERVAL:
        yolo_counter = 0
        
        # === Detecção com YOLO ===
        results = model.predict(
            source=frame, 
            conf=0.5,  
            verbose=False,
            save=False,  
            show=False,  
            device='cpu'
        )

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls] if hasattr(model, "names") else str(cls)
                conf = float(box.conf[0])
                if "fire" in label.lower() or "flame" in label.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.putText(annotated_frame, f"YOLO Fire {conf:.2f}", (x1, y1 - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    yolo_fire_detected = True

    # === Combinação das detecções ===
    combined = cv.addWeighted(result_hsv, 0.5, annotated_frame, 0.5, 0)
    if hsv_fire_detected or yolo_fire_detected:
        cv.putText(combined, "FOGO DETECTADO!", (40, 50), cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    # === OTIMIZAÇÃO: Redimensionamento otimizado ===
    # Só redimensiona se necessário
    if combined.shape[:2] != display_size[::-1]:  # [::-1] porque OpenCV usa (width, height)
        display_frame = cv.resize(combined, display_size)
    else:
        display_frame = combined
        
    cv.imshow("Detecção de Fogo (HSV + IA)", display_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()