import cv2 as cv
import sys
import numpy as np
from ultralytics import YOLO

class Camera:
    def __init__(self, index=1, size=(640, 480), fps=30, buffer_size=1):
        self.cap = cv.VideoCapture(index)
        if not self.cap.isOpened():
            sys.exit("Erro: não foi possível abrir o vídeo. Verifique a câmera e o índice.")
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv.CAP_PROP_FPS, fps)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, buffer_size)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            sys.exit("Erro: não foi possível ler o frame da câmera.")
        return frame

    def release(self):
        self.cap.release()

class BrightnessTracker:
    def __init__(self):
        self.last_brightness = 0.0

    def update(self, gray_frame):
        brightness = float(np.mean(gray_frame))
        flicker = abs(brightness - self.last_brightness)
        self.last_brightness = brightness
        return brightness, flicker

class HSVMasker:
    def __init__(self):
        self.lower_orange = np.array([10, 100, 180])
        self.upper_orange = np.array([35, 255, 255])
        self.lower_red1 = np.array([0, 150, 150])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 150, 150])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_white = np.array([0, 0, 220])
        self.upper_white = np.array([50, 60, 255])
        self.kernel = np.ones((3, 3), np.uint8)

    def make_mask(self, frame_bgr):
        hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
        hsv_blur = cv.GaussianBlur(hsv, (5, 5), 0)
        masks = [
            cv.inRange(hsv_blur, self.lower_orange, self.upper_orange),
            cv.inRange(hsv_blur, self.lower_red1, self.upper_red1),
            cv.inRange(hsv_blur, self.lower_red2, self.upper_red2),
            cv.inRange(hsv_blur, self.lower_white, self.upper_white),
        ]
        mask = masks[0]
        for m in masks[1:]:
            mask = cv.bitwise_or(mask, m)
        result_hsv = cv.bitwise_and(frame_bgr, frame_bgr, mask=mask)
        return mask, result_hsv

    def postprocess_mask(self, mask, result_hsv, min_pixels=100):
        if cv.countNonZero(mask) > min_pixels:
            edges = cv.morphologyEx(result_hsv, cv.MORPH_CLOSE, self.kernel)
            edges = cv.GaussianBlur(edges, (5, 5), 0)
            edges_gray = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
            edges_gray_canny = cv.Canny(edges_gray, 60, 160)
        else:
            edges_gray_canny = np.zeros(result_hsv.shape[:2], dtype=np.uint8)
        return edges_gray_canny

class HSVFireDetector:
    def __init__(self, min_area=100, flicker_threshold=2.0):
        self.min_area = min_area
        self.flicker_threshold = flicker_threshold

    def detect(self, edges_gray_canny, base_frame, flicker):
        detected = False
        annotated = base_frame.copy()
        contours, _ = cv.findContours(edges_gray_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv.contourArea(c)
            if area > self.min_area and flicker > self.flicker_threshold:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(annotated, "HSV Fire", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                detected = True
        return detected, annotated

class YOLODetector:
    def __init__(self, model_path="yolo_fire_mini.pt", interval=10, conf=0.5, device="cpu"):
        self.model = YOLO(model_path)
        self.interval = interval
        self.conf = conf
        self.device = device
        self.counter = 0

    def maybe_detect(self, frame_bgr):
        self.counter += 1
        if self.counter < self.interval:
            return False, frame_bgr
        self.counter = 0

        detected = False
        annotated = frame_bgr.copy()
        results = self.model.predict(
            source=frame_bgr, conf=self.conf, verbose=False, save=False, show=False, device=self.device
        )
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls] if hasattr(self.model, "names") else str(cls)
                conf = float(box.conf[0])
                if "fire" in label.lower() or "flame" in label.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.putText(annotated, f"YOLO Fire {conf:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    detected = True
        return detected, annotated

class Renderer:
    def __init__(self, display_size=(700, 500), window_name="Detecção de Fogo (HSV + IA)"):
        self.display_size = display_size
        self.window_name = window_name

    def compose(self, hsv_annotated, yolo_annotated, fire_flag):
        combined = cv.addWeighted(hsv_annotated, 0.5, yolo_annotated, 0.5, 0)
        if fire_flag:
            cv.putText(combined, "FOGO DETECTADO!", (40, 50), cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
        if combined.shape[:2] != self.display_size[::-1]:
            combined = cv.resize(combined, self.display_size)
        return combined

    def show(self, frame):
        cv.imshow(self.window_name, frame)

    def should_quit(self):
        return (cv.waitKey(1) & 0xFF) == ord('q')

class FireDetectionApp:
    def __init__(self):
        self.camera = Camera(index=1, size=(640, 480), fps=30, buffer_size=1)
        self.tracker = BrightnessTracker()
        self.masker = HSVMasker()
        self.hsv_detector = HSVFireDetector(min_area=100, flicker_threshold=2.0)
        self.yolo = YOLODetector(model_path="yolo_fire_mini.pt", interval=10, conf=0.5, device="cpu")
        self.renderer = Renderer(display_size=(700, 500), window_name="Detecção de Fogo (HSV + IA)")

    def run(self):
        try:
            while True:
                frame = self.camera.read()
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                _, flicker = self.tracker.update(gray)

                mask, result_hsv = self.masker.make_mask(frame)
                edges = self.masker.postprocess_mask(mask, result_hsv)

                hsv_detected, hsv_annot = self.hsv_detector.detect(edges, result_hsv, flicker)
                yolo_detected, yolo_annot = self.yolo.maybe_detect(frame)

                combined = self.renderer.compose(hsv_annot, yolo_annot, hsv_detected or yolo_detected)
                self.renderer.show(combined)

                if self.renderer.should_quit():
                    break
        finally:
            self.camera.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    FireDetectionApp().run()