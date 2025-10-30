import cv2 as cv
import sys
import numpy as np
from ultralytics import YOLO

# BLOCO 1 — ENTRADA DE VÍDEO (Câmera)
# Responsável por abrir, configurar e entregar frames.

class Camera:
    def __init__(self, index=1, size=(640, 480), fps=30, buffer_size=1):
   
        #index: índice da câmera (0, 1, ...).
        #size: resolução do frame (largura, altura).
        #fps: frames por segundo desejado.
        #buffer_size: tamanho do buffer interno. 1 ajuda reduzir atraso (latência).
     
        self.cap = cv.VideoCapture(index)
        if not self.cap.isOpened():
            sys.exit("Erro: não foi possível abrir o vídeo. Verifique a câmera e o índice.")
        # CAP_PROP_FRAME_WIDTH/HEIGHT: define resolução do frame que a câmera tentará fornecer
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, size[1])
        # CAP_PROP_FPS: taxa de quadros esperada (pode variar conforme a câmera/driver)
        self.cap.set(cv.CAP_PROP_FPS, fps)
        # CAP_PROP_BUFFERSIZE: quantidade de frames em fila; 1 reduz atraso
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, buffer_size)

    def read(self):
        #Lê um frame da câmera. Encerra o programa se falhar
        ret, frame = self.cap.read()
        if not ret:
            sys.exit("Erro: não foi possível ler o frame da câmera.")
        return frame

    def release(self):
        #Libera o dispositivo de captura.
        self.cap.release()


# BLOCO 2 — RASTREAMENTO DE BRILHO
# Mede brilho médio do frame e calcula variação (flicker).

class BrightnessTracker:
    def __init__(self):
        self.last_brightness = 0.0

    def update(self, gray_frame):

        #gray_frame: imagem em tons de cinza (1 canal).
        #brightness: média dos tons (0-255). Varia com iluminação.
        #flicker: |brightness_atual - brightness_anterior|.
        
        brightness = float(np.mean(gray_frame))
        flicker = abs(brightness - self.last_brightness)
        self.last_brightness = brightness
        return brightness, flicker



# BLOCO 3 — PRÉ-PROCESSAMENTO HSV (máscaras e bordas)
# Gera máscara de cores (laranja/vermelho/branco) e refina.

class HSVMasker:
    def __init__(self):
        # Faixas HSV heurísticas para tons comuns de fogo
        self.lower_orange = np.array([10, 100, 180])
        self.upper_orange = np.array([35, 255, 255])
        self.lower_red1 = np.array([0, 150, 150])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 150, 150])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_white = np.array([0, 0, 220])
        self.upper_white = np.array([50, 60, 255])
        # Kernel 3x3 para operações morfológicas (fechamento)
        self.kernel = np.ones((3, 3), np.uint8)

    def make_mask(self, frame_bgr):
        
        #Converte BGR->HSV, aplica blur e cria máscaras por faixa de cor.
        #Retorna: (mask binária, frame colorido filtrado pela mask).
        
        hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
        hsv_blur = cv.GaussianBlur(hsv, (5, 5), 0)  # reduz ruído pontual

        masks = [
            cv.inRange(hsv_blur, self.lower_orange, self.upper_orange),
            cv.inRange(hsv_blur, self.lower_red1, self.upper_red1),
            cv.inRange(hsv_blur, self.lower_red2, self.upper_red2),
            cv.inRange(hsv_blur, self.lower_white, self.upper_white),
        ]
        # bitwise_or combina as máscaras de cores
        mask = masks[0]
        for m in masks[1:]:
            mask = cv.bitwise_or(mask, m)

        # Aplica a máscara no frame original (mantém só regiões de interesse)
        result_hsv = cv.bitwise_and(frame_bgr, frame_bgr, mask=mask)
        return mask, result_hsv

    def postprocess_mask(self, mask, result_hsv, min_pixels=100):
        
        #Refina a máscara com fechamento, blur e Canny (bordas).
        #Só processa se houver pixels suficientes.
        #Retorna: imagem de bordas (1 canal).
        
        if cv.countNonZero(mask) > min_pixels:
            # MORPH_CLOSE (fechamento): preenche pequenos buracos na máscara
            edges = cv.morphologyEx(result_hsv, cv.MORPH_CLOSE, self.kernel)
            edges = cv.GaussianBlur(edges, (5, 5), 0)
            edges_gray = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
            # Canny: detector de bordas (limiares escolhidos empiricamente)
            edges_gray_canny = cv.Canny(edges_gray, 60, 160)
        else:
            # Se não há região suficiente, retorna "vazio"
            edges_gray_canny = np.zeros(result_hsv.shape[:2], dtype=np.uint8)
        return edges_gray_canny



# BLOCO 4 — DETECÇÃO POR HSV + CONTORNOS
# Analisa contornos nas bordas e decide se há fogo plausível.

class HSVFireDetector:
    def __init__(self, min_area=100, flicker_threshold=2.0):

        #min_area: área mínima do contorno para considerar (reduz falsos positivos).
        #flicker_threshold: exige variação de brilho mínima (fogo costuma oscilar).

        self.min_area = min_area
        self.flicker_threshold = flicker_threshold

    def detect(self, edges_gray_canny, base_frame, flicker):
        
        #edges_gray_canny: bordas (1 canal).
        #base_frame: frame colorido para anotações.
        #flicker: variação de brilho do frame atual.
        #Retorna: (bool detectado, frame anotado).

        detected = False
        annotated = base_frame.copy()
        # findContours encontra regiões conectadas nas bordas
        contours, _ = cv.findContours(edges_gray_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv.contourArea(c)
            if area > self.min_area and flicker > self.flicker_threshold:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(annotated, "HSV Fire", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                detected = True
        return detected, annotated



# BLOCO 5 — DETECÇÃO POR IA (YOLO)
# Roda o modelo YOLO a cada N frames para reduzir custo.

class YOLODetector:
    def __init__(self, model_path="yolo_fire_mini.pt", interval=10, conf=0.5, device="cpu"):
     
        #model_path: caminho do .pt treinado (ou YOLO genérico).
        #interval: roda YOLO a cada N frames (economia de processamento).
        #conf: confiança mínima das detecções.
        #device: 'cpu' ou 'cuda' se houver GPU.
   
        self.model = YOLO(model_path)  # carrega o modelo
        self.interval = interval
        self.conf = conf
        self.device = device
        self.counter = 0

    def maybe_detect(self, frame_bgr):

        #Executa YOLO apenas quando o contador atingir 'interval'.
        #Retorna: (bool detectado, frame anotado).

        self.counter += 1
        if self.counter < self.interval:
            return False, frame_bgr
        self.counter = 0

        detected = False
        annotated = frame_bgr.copy()

        # model.predict:
        # - source: imagem/frame
        # - conf: limiar de confiança
        # - verbose=False: sem logs detalhados
        # - save=False/show=False: não salvar/exibir janelas do YOLO
        # - device: em qual dispositivo rodar ('cpu' ou 'cuda')
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            verbose=False,
            save=False,
            show=False,
            device=self.device
        )

        # results contém lista de resultados; cada r possui r.boxes com as detecções
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  # índice da classe
                label = self.model.names[cls] if hasattr(self.model, "names") else str(cls)
                conf = float(box.conf[0])  # confiança do box
                # Ajuste o filtro conforme as classes do seu modelo (.names)
                if "fire" in label.lower() or "flame" in label.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordenadas do retângulo
                    cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.putText(annotated, f"YOLO Fire {conf:.2f}", (x1, y1 - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    detected = True
        return detected, annotated


# BLOCO 6 — RENDERIZAÇÃO/EXIBIÇÃO
# Mistura camadas e exibe. Centraliza tamanho e janela.

class Renderer:
    def __init__(self, display_size=(700, 500), window_name="Detecção de Fogo (HSV + IA)"):
       
        #display_size: resolução final exibida na janela (w, h).
        #window_name: título da janela do OpenCV.
        
        self.display_size = display_size
        self.window_name = window_name

    def compose(self, hsv_annotated, yolo_annotated, fire_flag):
        
        #Faz um blend 50/50 entre a camada HSV e a camada YOLO (detecção dupla)
        #Se houver detecção, escreve um aviso na imagem.
       
        # addWeighted: mistura as duas imagens com pesos iguais
        combined = cv.addWeighted(hsv_annotated, 0.5, yolo_annotated, 0.5, 0)
        if fire_flag:
            cv.putText(combined, "FOGO DETECTADO!", (40, 50),
                       cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
        # Redimensiona se necessário (OpenCV usa (w, h) na função resize)
        if combined.shape[:2] != self.display_size[::-1]:
            combined = cv.resize(combined, self.display_size)
        return combined

    def show(self, frame):
        #Exibe o frame em uma janela
        cv.imshow(self.window_name, frame)

    def should_quit(self):
 
        #Fecha ao pressionar 'q'.
        #cv.waitKey(1) lê teclado; & 0xFF normaliza o valor da tecla.
    
        return (cv.waitKey(1) & 0xFF) == ord('q')



# BLOCO 7 — APLICAÇÃO
# Liga todos os componentes dentro de um loop principal.

class FireDetectionApp:
    def __init__(self):
        # Configurações agrupadas aqui para rápido ajuste
        self.camera = Camera(index=1, size=(640, 480), fps=30, buffer_size=1)
        self.tracker = BrightnessTracker()
        self.masker = HSVMasker()
        self.hsv_detector = HSVFireDetector(min_area=100, flicker_threshold=2.0)
        self.yolo = YOLODetector(model_path="yolo_fire_mini.pt", interval=10, conf=0.5, device="cpu")
        self.renderer = Renderer(display_size=(700, 500), window_name="Detecção de Fogo (HSV + IA)")

    def run(self):
        """
        Loop:
          1) Lê frame
          2) Atualiza brilho/flicker
          3) Gera máscara HSV e bordas
          4) Detecta fogo por HSV
          5) Detecta fogo por YOLO (por intervalo)
          6) Faz blend e exibe
        """
        try:
            while True:
                # 1) Leitura do frame
                frame = self.camera.read()

                # 2) Brilho/Flicker (trabalha em cinza para ser barato)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                _, flicker = self.tracker.update(gray)

                # 3) Máscara HSV + pós-processamento
                mask, result_hsv = self.masker.make_mask(frame)
                edges = self.masker.postprocess_mask(mask, result_hsv)

                # 4) Detecção por HSV
                hsv_detected, hsv_annot = self.hsv_detector.detect(edges, result_hsv, flicker)

                # 5) Detecção por YOLO (talvez neste frame)
                yolo_detected, yolo_annot = self.yolo.maybe_detect(frame)

                # 6) Composição/Exibição
                combined = self.renderer.compose(hsv_annot, yolo_annot, hsv_detected or yolo_detected)
                self.renderer.show(combined)

                if self.renderer.should_quit():
                    break
        finally:
            # Fecha recursos de forma segura mesmo em caso de erro
            self.camera.release()
            cv.destroyAllWindows()


# Ponto de entrada do script
if __name__ == "__main__":
    FireDetectionApp().run()