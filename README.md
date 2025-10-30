## Fire Detection (OpenCV + YOLO, POO em Python)

Implementação de detecção de fogo com foco em Programação Orientada a Objetos (POO). O código principal é o `fire_detec_oop.py`, organizado em classes.

### Principais classes no `fire_detec_oop.py`
- **Camera**: gerencia a captura de vídeo (abrir, ler, liberar).
- **BrightnessTracker**: calcula brilho médio do frame e variação (flicker).
- **HSVMasker**: pré-processa o frame (HSV, blur, máscaras de cor, bordas).
- **HSVFireDetector**: usa contornos e flicker para decidir se há fogo plausível.
- **YOLODetector**: roda YOLO a cada N frames (intervalo) para confirmar detecções.
- **Renderer**: combina camadas HSV+YOLO, escreve aviso e exibe na janela.
- **FireDetectionApp**: orquestra o fluxo no loop principal.

### Como funciona (resumo)
1. Captura frame da câmera.  
2. Calcula brilho e flicker.  
3. Gera máscara por faixas HSV (tons de fogo), refina e extrai bordas.  
4. Detecta regiões plausíveis por contornos (HSV).  
5. Roda YOLO periodicamente para detectar “fire/flame”.  
6. Faz blend HSV+YOLO e exibe; tecla `q` encerra.

### Requisitos
- Python 3.8+
- OpenCV, NumPy, ultralytics (YOLO)

```bash
pip install opencv-python numpy ultralytics
```

- Modelo YOLO (.pt) disponível localmente, por padrão `yolo_fire_mini.pt` na raiz do projeto.  
- Câmera acessível no índice configurado (padrão: 1).

### Executar
```bash
python fire_detec_oop.py
```

### Parâmetros principais (editáveis no código)
- `Camera(index=1, size=(640, 480), fps=30)`
- `YOLODetector(model_path="yolo_fire_mini.pt", interval=10, conf=0.5, device="cpu")`
- `HSVFireDetector(min_area=100, flicker_threshold=2.0)`
- `Renderer(display_size=(700, 500))`

### Outras versões no repositório
- `Deteccao.py`  
  - Versão procedural (loop único) com a mesma lógica base (HSV + contornos + YOLO por intervalo).  
  - Útil para comparar antes/depois da POO.
- `capture_fire.py`  
  - Versão mínima só com HSV (sem YOLO, sem organização em classes).  
  - Boa para testes rápidos de máscara e contornos.

### Notas
- Se o modelo YOLO tiver rótulos diferentes de “fire/flame”, ajuste o filtro em `YOLODetector.maybe_detect` (condição no `label.lower()`).
- O índice da câmera (`index=1`) pode variar. Se necessário, altere para `0`.
- O `interval` do YOLO regula desempenho (maior = mais leve; menor = mais responsivo).